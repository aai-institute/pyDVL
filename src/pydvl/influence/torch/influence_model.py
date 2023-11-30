import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from math import prod
from typing import Callable, Optional

import torch
from torch import nn as nn
from torch.utils.data import DataLoader

from ...utils import maybe_progress
from ...utils.progress import log_duration
from ..base_influence_model import (
    Influence,
    InfluenceType,
    UnSupportedInfluenceTypeException,
)
from ..inversion import InfluenceRegistry, InversionMethod
from .functional import (
    get_batch_hvp,
    get_hessian,
    get_hvp_function,
    matrix_jacobian_product,
    per_sample_gradient,
    per_sample_mixed_derivative,
)
from .torch_differentiable import (
    LowRankProductRepresentation,
    TorchTwiceDifferentiable,
    model_hessian_low_rank,
    solve_cg,
)
from .util import flatten_dimensions

logger = logging.getLogger(__name__)


class TorchInfluence(Influence[torch.Tensor], ABC):
    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        self.loss = loss
        self.model = model
        self._num_parameters = sum(
            [p.numel() for p in model.parameters() if p.requires_grad]
        )
        self._model_device = next(
            (p.device for p in model.parameters() if p.requires_grad)
        )
        self._model_params = {
            k: p.detach() for k, p in self.model.named_parameters() if p.requires_grad
        }
        super().__init__()

    @property
    def num_parameters(self):
        return self._num_parameters

    @property
    def model_device(self):
        return self._model_device

    @property
    def model_params(self):
        return self._model_params

    @log_duration
    def _loss_grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        grads = per_sample_gradient(self.model, self.loss)(self.model_params, x, y)
        shape = (x.shape[0], -1)
        return flatten_dimensions(grads.values(), shape=shape)

    @log_duration
    def _flat_loss_mixed_grad(self, x: torch.Tensor, y: torch.Tensor):
        mixed_grads = per_sample_mixed_derivative(self.model, self.loss)(
            self.model_params, x, y
        )
        shape = (x.shape[0], prod(x.shape[1:]), -1)
        return flatten_dimensions(mixed_grads.values(), shape=shape)

    def values(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        influence_type: InfluenceType = InfluenceType.Up,
    ) -> torch.Tensor:

        if x is None:

            if y is not None:
                raise ValueError(
                    "Providing labels y, without providing model input x is not supported"
                )

            return self._symmetric_values(
                x_test.to(self.model_device),
                y_test.to(self.model_device),
                influence_type,
            )

        if y is None:
            raise ValueError(
                "Providing model input x without providing labels y is not supported"
            )

        return self._non_symmetric_values(
            x_test.to(self.model_device),
            y_test.to(self.model_device),
            x.to(self.model_device),
            y.to(self.model_device),
            influence_type,
        )

    def _non_symmetric_values(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        influence_type: InfluenceType = InfluenceType.Up,
    ):
        if influence_type == InfluenceType.Up:
            if x_test.shape[0] <= x.shape[0]:
                factor = self.factors(x_test, y_test)
                values = self.values_from_factors(
                    factor, x, y, influence_type=influence_type
                )
            else:
                factor = self.factors(x, y)
                values = self.values_from_factors(
                    factor, x_test, y_test, influence_type=influence_type
                ).T
        elif influence_type == InfluenceType.Perturbation:
            factor = self.factors(x_test, y_test)
            values = self.values_from_factors(
                factor, x, y, influence_type=influence_type
            )
        else:
            raise UnSupportedInfluenceTypeException(influence_type)
        return values

    def _symmetric_values(
        self, x: torch.Tensor, y: torch.Tensor, influence_type: InfluenceType
    ) -> torch.Tensor:

        grad = self._loss_grad(x, y)
        fac = self._solve_hvp(grad)

        if influence_type == InfluenceType.Up:
            values = fac @ grad.T
        elif influence_type == InfluenceType.Perturbation:
            values = self.values_from_factors(fac, x, y, influence_type=influence_type)
        else:
            raise UnSupportedInfluenceTypeException(influence_type)
        return values

    def factors(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        return self._solve_hvp(
            self._loss_grad(x.to(self.model_device), y.to(self.model_device))
        )

    def values_from_factors(
        self,
        z_test_factors: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        influence_type: InfluenceType = InfluenceType.Up,
    ) -> torch.Tensor:
        if influence_type == InfluenceType.Up:
            return (
                z_test_factors
                @ self._loss_grad(x.to(self.model_device), y.to(self.model_device)).T
            )
        elif influence_type == InfluenceType.Perturbation:
            return torch.einsum(
                "ia,jba->ijb",
                z_test_factors,
                self._flat_loss_mixed_grad(
                    x.to(self.model_device), y.to(self.model_device)
                ),
            )
        else:
            raise UnSupportedInfluenceTypeException(influence_type)

    @abstractmethod
    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        pass


class DirectInfluence(TorchInfluence):
    r"""
    Given a model and training data, it finds x such that \(Hx = b\), with \(H\) being the model hessian.

    Args:
        model: instance of [torch.nn.Module][torch.nn.Module].
        train_dataloader: A DataLoader containing the training data.
        hessian_regularization: Regularization of the hessian.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        hessian_regularization: float,
        hessian: torch.Tensor = None,
        train_dataloader: DataLoader = None,
    ):
        if hessian is None and train_dataloader is None:
            raise ValueError(
                f"Either provide a pre-computed hessian or a data_loader to compute the hessian"
            )

        super().__init__(model, loss)
        self.hessian_regularization = hessian_regularization
        self.hessian = (
            hessian
            if hessian is not None
            else get_hessian(model, loss, train_dataloader)
        )

    @log_duration
    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(
            self.hessian.to(self.model_device)
            + self.hessian_regularization
            * torch.eye(self.num_parameters, device=self.model_device),
            rhs.T.to(self.model_device),
        ).T

    def to(self, device: torch.device):
        self.hessian = self.hessian.to(device)
        self.model = self.model.to(device)
        self._model_device = device
        self._model_params = {
            k: p.detach().to(device)
            for k, p in self.model.named_parameters()
            if p.requires_grad
        }
        return self


class BatchCgInfluence(TorchInfluence):
    r"""
    Given a model and training data, it uses conjugate gradient to calculate the
    inverse of the Hessian Vector Product. More precisely, it finds x such that \(Hx =
    b\), with \(H\) being the model hessian. For more info, see
    [Wikipedia](https://en.wikipedia.org/wiki/Conjugate_gradient_method).

    Args:
        model: instance of [torch.nn.Module][torch.nn.Module].
        A callable that takes the model's output and target as input and returns the scalar loss.
        train_dataloader: A DataLoader containing the training data.
        hessian_regularization: Regularization of the hessian.
        x0: Initial guess for hvp. If None, defaults to b.
        rtol: Maximum relative tolerance of result.
        atol: Absolute tolerance of result.
        maxiter: Maximum number of iterations. If None, defaults to 10*len(b).
        progress: If True, display progress bars.

    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_dataloader: DataLoader,
        hessian_regularization: float,
        x0: Optional[torch.Tensor] = None,
        rtol: float = 1e-7,
        atol: float = 1e-7,
        maxiter: Optional[int] = None,
        progress: bool = False,
    ):
        super().__init__(model, loss)
        self.progress = progress
        self.maxiter = maxiter
        self.atol = atol
        self.rtol = rtol
        self.x0 = x0
        self.hessian_regularization = hessian_regularization
        self.train_dataloader = train_dataloader

    @log_duration
    def values(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        influence_type: InfluenceType = InfluenceType.Up,
    ) -> torch.Tensor:
        r"""
        Compute approximation of

        \[ \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}}, f_{\theta}(x_{\text{test}})),
            \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle, \]

        for the case of up-weighting influence, resp.

        \[ \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}}, f_{\theta}(x_{\text{test}})),
            \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the perturbation type influence case. The approximate action of $H^{-1}$ is achieved via the
        [conjugate gradient method](https://en.wikipedia.org/wiki/Conjugate_gradient_method).

        Args:
            x_test: model input to use in the gradient computations of $H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
                f_{\theta}(x_{\text{test}}))$
            y_test: label tensor to compute gradients
            x: optional model input to use in the gradient computations $\nabla_{\theta}\ell(y, f_{\theta}(x))$,
               resp. $\nabla_{x}\nabla_{\theta}\ell(y, f_{\theta}(x))$, if None, use $x=x_{\text{test}}$
            y: optional label tensor to compute gradients
            influence_type: enum value of [InfluenceType][pydvl.influence.twice_differentiable.InfluenceType]

        Returns:
            [torch.nn.Tensor][torch.nn.Tensor] representing the element-wise scalar products for the provided batch.

        """
        return super().values(x_test, y_test, x, y, influence_type=influence_type)

    @log_duration
    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        if len(self.train_dataloader) == 0:
            raise ValueError("Training dataloader must not be empty.")

        hvp = get_hvp_function(self.model, self.loss, self.train_dataloader)
        reg_hvp = lambda v: hvp(v) + self.hessian_regularization * v.type(rhs.dtype)
        batch_cg = torch.zeros_like(rhs)

        for idx, bi in enumerate(
            maybe_progress(rhs, self.progress, desc="Conjugate gradient")
        ):
            batch_result, batch_info = solve_cg(
                reg_hvp,
                bi,
                x0=self.x0,
                rtol=self.rtol,
                atol=self.atol,
                maxiter=self.maxiter,
            )
            batch_cg[idx] = batch_result
        return batch_cg

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        self._model_params = {
            k: p.detach().to(device)
            for k, p in self.model.named_parameters()
            if p.requires_grad
        }
        self._model_device = device
        return self


class LissaInfluence(TorchInfluence):
    r"""
    Uses LISSA, Linear time Stochastic Second-Order Algorithm, to iteratively
    approximate the inverse Hessian. More precisely, it finds x s.t. \(Hx = b\),
    with \(H\) being the model's second derivative wrt. the parameters.
    This is done with the update

    \[H^{-1}_{j+1} b = b + (I - d) \ H - \frac{H^{-1}_j b}{s},\]

    where \(I\) is the identity matrix, \(d\) is a dampening term and \(s\) a scaling
    factor that are applied to help convergence. For details, see
    (Koh and Liang, 2017)<sup><a href="#koh_liang_2017">1</a></sup> and the original paper
    (Agarwal et al.)<sup><a href="#agarwal_secondorder_2017">2</a></sup>.

    Args:
        model: instance of [torch.nn.Module][torch.nn.Module].
        train_dataloader: A DataLoader containing the training data.
        hessian_regularization: Regularization of the hessian.
        maxiter: Maximum number of iterations.
        dampen: Dampening factor, defaults to 0 for no dampening.
        scale: Scaling factor, defaults to 10.
        h0: Initial guess for hvp.
        rtol: tolerance to use for early stopping
        progress: If True, display progress bars.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_dataloader: DataLoader,
        hessian_regularization: float,
        maxiter: int = 1000,
        dampen: float = 0.0,
        scale: float = 10.0,
        h0: Optional[torch.Tensor] = None,
        rtol: float = 1e-4,
        progress: bool = False,
    ):
        super().__init__(model, loss)
        self.maxiter = maxiter
        self.hessian_regularization = hessian_regularization
        self.progress = progress
        self.rtol = rtol
        self.h0 = h0
        self.scale = scale
        self.dampen = dampen
        self.train_dataloader = train_dataloader

    @log_duration
    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:

        h_estimate = self.h0 if self.h0 is not None else torch.clone(rhs)

        shuffled_training_data = DataLoader(
            self.train_dataloader.dataset,
            self.train_dataloader.batch_size,
            shuffle=True,
        )

        def lissa_step(
            h: torch.Tensor, reg_hvp: Callable[[torch.Tensor], torch.Tensor]
        ) -> torch.Tensor:
            """Given an estimate of the hessian inverse and the regularised hessian
            vector product, it computes the next estimate.

            Args:
                h: An estimate of the hessian inverse.
                reg_hvp: Regularised hessian vector product.

            Returns:
                The next estimate of the hessian inverse.
            """
            return rhs + (1 - self.dampen) * h - reg_hvp(h) / self.scale

        model_params = {
            k: p.detach() for k, p in self.model.named_parameters() if p.requires_grad
        }
        b_hvp = torch.vmap(
            get_batch_hvp(self.model, self.loss), in_dims=(None, None, None, 0)
        )
        for _ in maybe_progress(range(self.maxiter), self.progress, desc="Lissa"):
            x, y = next(iter(shuffled_training_data))
            # grad_xy = model.grad(x, y, create_graph=True)
            reg_hvp = (
                lambda v: b_hvp(model_params, x, y, v) + self.hessian_regularization * v
            )
            residual = lissa_step(h_estimate, reg_hvp) - h_estimate
            h_estimate += residual
            if torch.isnan(h_estimate).any():
                raise RuntimeError("NaNs in h_estimate. Increase scale or dampening.")
            max_residual = torch.max(torch.abs(residual / h_estimate))
            if max_residual < self.rtol:
                break

        mean_residual = torch.mean(torch.abs(residual / h_estimate))

        logger.info(
            f"Terminated Lissa with {max_residual*100:.2f} % max residual."
            f" Mean residual: {mean_residual*100:.5f} %"
        )
        return h_estimate / self.scale


class ArnoldiInfluence(TorchInfluence):
    r"""
    Solves the linear system Hx = b, where H is the Hessian of the model's loss function and b is the given
    right-hand side vector.
    It employs the [implicitly restarted Arnoldi method](https://en.wikipedia.org/wiki/Arnoldi_iteration) for
    computing a partial eigen decomposition, which is used fo the inversion i.e.

    \[x = V D^{-1} V^T b\]

    where \(D\) is a diagonal matrix with the top (in absolute value) `rank_estimate` eigenvalues of the Hessian
    and \(V\) contains the corresponding eigenvectors.

    Args:
        model: Instance of [torch.nn.Module][torch.nn.Module].
            The Hessian will be calculated with respect to this model's parameters.
        train_dataloader: A DataLoader instance that provides the model's training data.
            Used in calculating the Hessian-vector products.
        hessian_regularization: Optional regularization parameter added to the Hessian-vector
            product for numerical stability.
        rank_estimate: The number of eigenvalues and corresponding eigenvectors to compute.
            Represents the desired rank of the Hessian approximation.
        krylov_dimension: The number of Krylov vectors to use for the Lanczos method.
            Defaults to min(model's number of parameters, max(2 times rank_estimate + 1, 20)).
        low_rank_representation: An instance of
            [LowRankProductRepresentation][pydvl.influence.torch.torch_differentiable.LowRankProductRepresentation]
            containing a previously computed low-rank representation of the Hessian. If provided, all other parameters
            are ignored; otherwise, a new low-rank representation is computed
            using provided parameters.
        tol: The stopping criteria for the Lanczos algorithm.
            Ignored if `low_rank_representation` is provided.
        max_iter: The maximum number of iterations for the Lanczos method.
            Ignored if `low_rank_representation` is provided.
        eigen_computation_on_gpu: If True, tries to execute the eigen pair approximation on the model's device
            via a cupy implementation. Ensure the model size or rank_estimate is appropriate for device memory.
            If False, the eigen pair approximation is executed on the CPU by the scipy wrapper to ARPACK.
    """

    def __init__(
        self,
        model,
        loss,
        low_rank_representation: Optional[LowRankProductRepresentation] = None,
        train_dataloader: DataLoader = None,
        hessian_regularization: float = 0.0,
        rank_estimate: int = 10,
        krylov_dimension: Optional[int] = None,
        tol: float = 1e-6,
        max_iter: Optional[int] = None,
        eigen_computation_on_gpu: bool = False,
    ):
        if low_rank_representation is None and train_dataloader is None:
            raise ValueError(
                f"Either provide a pre-computed hessian or a data_loader to compute the hessian"
            )

        if low_rank_representation is None:
            low_rank_representation = model_hessian_low_rank(
                model,
                loss,
                train_dataloader,
                hessian_perturbation=0.0,  # regularization is applied, when computing values
                rank_estimate=rank_estimate,
                krylov_dimension=krylov_dimension,
                tol=tol,
                max_iter=max_iter,
                eigen_computation_on_gpu=eigen_computation_on_gpu,
            )

        super().__init__(model, loss)
        self.low_rank_representation = low_rank_representation.to(self.model_device)
        self.hessian_regularization = hessian_regularization

    def _non_symmetric_values(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        influence_type: InfluenceType = InfluenceType.Up,
    ) -> torch.Tensor:

        if influence_type == InfluenceType.Up:
            mjp = matrix_jacobian_product(
                self.model, self.loss, self.low_rank_representation.projections.T
            )
            left = mjp(self.model_params, x_test, y_test)

            regularized_eigenvalues = (
                self.low_rank_representation.eigen_vals + self.hessian_regularization
            )

            right = torch.diag_embed(1.0 / regularized_eigenvalues) @ mjp(
                self.model_params, x, y
            )
            values = torch.einsum("ij, ik -> jk", left, right)
        elif influence_type == InfluenceType.Perturbation:
            factors = self.factors(x_test, y_test)
            values = self.values_from_factors(
                factors, x, y, influence_type=influence_type
            )
        else:
            raise UnSupportedInfluenceTypeException(influence_type)
        return values

    def _symmetric_values(
        self, x: torch.Tensor, y: torch.Tensor, influence_type: InfluenceType
    ) -> torch.Tensor:

        if influence_type == InfluenceType.Up:
            left = matrix_jacobian_product(
                self.model, self.loss, self.low_rank_representation.projections.T
            )(self.model_params, x, y)
            regularized_eigenvalues = (
                self.low_rank_representation.eigen_vals + self.hessian_regularization
            )
            right = torch.diag_embed(1.0 / regularized_eigenvalues) @ left
            values = torch.einsum("ij, ik -> jk", left, right)
        elif influence_type == InfluenceType.Perturbation:
            factors = self.factors(x, y)
            values = self.values_from_factors(
                factors, x, y, influence_type=influence_type
            )
        else:
            raise UnSupportedInfluenceTypeException(influence_type)
        return values

    @log_duration
    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:

        regularized_eigenvalues = (
            self.low_rank_representation.eigen_vals + self.hessian_regularization
        )

        result = self.low_rank_representation.projections @ (
            torch.diag_embed(1.0 / regularized_eigenvalues)
            @ (self.low_rank_representation.projections.t() @ rhs.t())
        )

        return result.t()

    def to(self, device: torch.device):
        return ArnoldiInfluence(
            self.model.to(device), self.loss, self.low_rank_representation.to(device)
        )


@InfluenceRegistry.register(TorchTwiceDifferentiable, InversionMethod.Direct)
def direct_factory(
    twice_differentiable: TorchTwiceDifferentiable,
    data_loader: DataLoader,
    hessian_regularization: float,
    **kwargs,
):
    return DirectInfluence(
        twice_differentiable.model,
        twice_differentiable.loss,
        train_dataloader=data_loader,
        hessian_regularization=hessian_regularization,
        **kwargs,
    )


@InfluenceRegistry.register(TorchTwiceDifferentiable, InversionMethod.Cg)
def cg_factory(
    twice_differentiable: TorchTwiceDifferentiable,
    data_loader: DataLoader,
    hessian_regularization: float,
    **kwargs,
):
    return BatchCgInfluence(
        twice_differentiable.model,
        twice_differentiable.loss,
        train_dataloader=data_loader,
        hessian_regularization=hessian_regularization,
        **kwargs,
    )


@InfluenceRegistry.register(TorchTwiceDifferentiable, InversionMethod.Lissa)
def lissa_factory(
    twice_differentiable: TorchTwiceDifferentiable,
    data_loader: DataLoader,
    hessian_regularization: float,
    **kwargs,
):
    return LissaInfluence(
        twice_differentiable.model,
        twice_differentiable.loss,
        train_dataloader=data_loader,
        hessian_regularization=hessian_regularization,
        **kwargs,
    )


@InfluenceRegistry.register(TorchTwiceDifferentiable, InversionMethod.Arnoldi)
def arnoldi_factory(
    twice_differentiable: TorchTwiceDifferentiable,
    data_loader: DataLoader,
    hessian_regularization: float,
    **kwargs,
):
    return ArnoldiInfluence(
        twice_differentiable.model,
        twice_differentiable.loss,
        train_dataloader=data_loader,
        hessian_regularization=hessian_regularization,
        **kwargs,
    )

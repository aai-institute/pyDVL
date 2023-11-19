import logging
from abc import ABC, abstractmethod
from math import prod
from typing import Callable, Optional

import torch
from torch import nn as nn
from torch.utils.data import DataLoader

from ...utils import maybe_progress
from ..inversion import InfluenceRegistry, InversionMethod
from ..twice_differentiable import Influence, InfluenceType, InverseHvpResult
from .functional import (
    get_batch_hvp,
    get_hessian,
    matrix_jacobian_product,
    per_sample_gradient,
    per_sample_mixed_derivative,
)
from .torch_differentiable import (
    LowRankProductRepresentation,
    TorchTwiceDifferentiable,
    model_hessian_low_rank,
    solve_arnoldi,
    solve_batch_cg,
    solve_lissa,
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

    def _loss_grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        grads = per_sample_gradient(self.model, self.loss)(self.model_params, x, y)
        shape = (x.shape[0], -1)
        return flatten_dimensions(grads.values(), shape=shape)

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
    ) -> InverseHvpResult:

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
            raise ValueError()  # TODO: error message

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
        y_test: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        influence_type: InfluenceType = InfluenceType.Up,
    ):

        if influence_type is InfluenceType.Up:
            if x_test.shape[0] <= y.shape[0]:
                factor, info = self.factors(x_test, y_test)
                values = self.up_weighting(factor, x, y)
            else:
                factor, info = self.factors(x, y)
                values = self.up_weighting(factor, x_test, y_test)
        else:
            factor, info = self.factors(x_test, y_test)
            values = self.perturbation(factor, x, y)
        return InverseHvpResult(values, info)

    def _symmetric_values(
        self, x: torch.Tensor, y: torch.Tensor, influence_type: InfluenceType
    ) -> InverseHvpResult[torch.Tensor]:

        grad = self._loss_grad(x, y)
        fac, info = self._solve_hvp(grad)

        if influence_type is InfluenceType.Up:
            values = fac @ grad.T
        else:
            values = self.perturbation(fac, x, y)

        return InverseHvpResult(values, info)

    def up_weighting(
        self,
        z_test_factors: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:

        return (
            z_test_factors
            @ self._loss_grad(x.to(self.model_device), y.to(self.model_device)).T
        )

    def perturbation(
        self,
        z_test_factors: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:

        return torch.einsum(
            "ia,jba->ijb",
            z_test_factors,
            self._flat_loss_mixed_grad(
                x.to(self.model_device), y.to(self.model_device)
            ),
        )

    def factors(self, x: torch.Tensor, y: torch.Tensor) -> InverseHvpResult:

        return self._solve_hvp(
            self._loss_grad(x.to(self.model_device), y.to(self.model_device))
        )

    @abstractmethod
    def _solve_hvp(self, rhs: torch.Tensor) -> InverseHvpResult:
        pass


class DirectInfluence(TorchInfluence):
    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        hessian_regularization: float,
        hessian: torch.Tensor = None,
        train_dataloader: DataLoader = None,
        return_hessian_in_info: bool = False,
    ):
        if hessian is None and train_dataloader is None:
            raise ValueError(
                f"Either provide a pre-computed hessian or a data_loader to compute the hessian"
            )

        super().__init__(model, loss)
        self.return_hessian_in_info = return_hessian_in_info
        self.hessian_perturbation = hessian_regularization
        self.hessian = (
            hessian
            if hessian is not None
            else get_hessian(model, loss, train_dataloader)
        )

    def prepare_for_distributed(self) -> "Influence":
        if self.return_hessian_in_info:
            self.return_hessian_in_info = False
            logger.warning(
                f"Modified parameter `return_hessian_in_info` to `False`, "
                f"to prepare for distributed computing"
            )
        return self

    def _solve_hvp(self, rhs: torch.Tensor) -> InverseHvpResult:
        result = torch.linalg.solve(
            self.hessian.to(self.model_device)
            + self.hessian_perturbation
            * torch.eye(self.num_parameters, device=self.model_device),
            rhs.T.to(self.model_device),
        ).T
        info = {}
        if self.return_hessian_in_info:
            info["hessian"] = self.hessian
        return InverseHvpResult(result, info)

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

    def _solve_hvp(self, rhs: torch.Tensor) -> InverseHvpResult:
        # TODO directly implement the method here and remove call to obsolete function
        x, info = solve_batch_cg(
            TorchTwiceDifferentiable(self.model, self.loss),
            self.train_dataloader,
            rhs,
            self.hessian_regularization,
            x0=self.x0,
            rtol=self.rtol,
            atol=self.atol,
            maxiter=self.maxiter,
        )
        return InverseHvpResult(x, info)

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
    """
    Uses LISSA, Linear time Stochastic Second-Order Algorithm, to iteratively
    approximate the inverse Hessian. More precisely, it finds x s.t. \(Hx = b\),
    with \(H\) being the model's second derivative wrt. the parameters.
    This is done with the update

    \[H^{-1}_{j+1} b = b + (I - d) \ H - \frac{H^{-1}_j b}{s},\]

    where \(I\) is the identity matrix, \(d\) is a dampening term and \(s\) a scaling
    factor that are applied to help convergence. For details, see
    (Koh and Liang, 2017)<sup><a href="#koh_liang_2017">1</a></sup> and the original paper
    (Agarwal et. al.)<sup><a href="#agarwal_secondorder_2017">2</a></sup>.

    Args:
        model: instance of [torch.nn.Module][torch.nn.Module].
        training_data: A DataLoader containing the training data.
        hessian_perturbation: Regularization of the hessian.
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

    def _solve_hvp(self, rhs: torch.Tensor) -> InverseHvpResult:

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
        info = {
            "max_perc_residual": max_residual * 100,
            "mean_perc_residual": mean_residual * 100,
        }
        return InverseHvpResult(x=h_estimate / self.scale, info=info)


class ArnoldiInfluence(TorchInfluence):
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
        return_low_rank_representation_in_info: bool = False,
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
                hessian_perturbation=hessian_regularization,
                rank_estimate=rank_estimate,
                krylov_dimension=krylov_dimension,
                tol=tol,
                max_iter=max_iter,
                eigen_computation_on_gpu=eigen_computation_on_gpu,
            )

        super().__init__(model, loss)
        self.low_rank_representation = low_rank_representation.to(self.model_device)
        self.return_low_rank_representation_in_info = (
            return_low_rank_representation_in_info
        )

    def prepare_for_distributed(self) -> "Influence":
        if self.return_low_rank_representation_in_info:
            self.return_low_rank_representation_in_info = False
            logger.warning(
                f"Modified parameter `return_low_rank_representation_in_info` to `False`, "
                f"to prepare for distributed computing"
            )
        return self

    def _non_symmetric_values(
        self,
        x_test: torch.Tensor,
        y_test: Optional[torch.Tensor] = None,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        influence_type: InfluenceType = InfluenceType.Up,
    ) -> InverseHvpResult[torch.Tensor]:

        if influence_type is InfluenceType.Up:
            mjp = matrix_jacobian_product(
                self.model, self.loss, self.low_rank_representation.projections.T
            )
            left = mjp(self.model_params, x_test, y_test)
            right = torch.diag_embed(
                1.0 / self.low_rank_representation.eigen_vals
            ) @ mjp(self.model_params, x, y)
            values = torch.einsum("ij, ik -> jk", left, right)
        else:
            factors, _ = self.factors(x_test, y_test)
            values = self.perturbation(factors, x, y)
        info = {}
        if self.return_low_rank_representation_in_info:
            info["low_rank_representation"] = self.low_rank_representation
        return InverseHvpResult(values, info)

    def _symmetric_values(
        self, x: torch.Tensor, y: torch.Tensor, influence_type: InfluenceType
    ) -> InverseHvpResult[torch.Tensor]:

        if influence_type is InfluenceType.Up:
            left = matrix_jacobian_product(
                self.model, self.loss, self.low_rank_representation.projections.T
            )(self.model_params, x, y)
            right = (
                torch.diag_embed(1.0 / self.low_rank_representation.eigen_vals) @ left
            )
            values = torch.einsum("ij, ik -> jk", left, right)
        else:
            factors, _ = self.factors(x, y)
            values = self.perturbation(factors, x, y)
        info = {}
        if self.return_low_rank_representation_in_info:
            info["low_rank_representation"] = self.low_rank_representation
        return InverseHvpResult(values, info)

    def _solve_hvp(self, rhs: torch.Tensor) -> InverseHvpResult:
        # TODO directly implement the method here and remove call to obsolete function
        x, info = solve_arnoldi(
            TorchTwiceDifferentiable(self.model, self.loss),
            None,
            rhs,
            low_rank_representation=self.low_rank_representation,
        )
        if not self.return_low_rank_representation_in_info:
            info = {}
        return InverseHvpResult(x, info)

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
        return_hessian_in_info=True,
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
        return_low_rank_representation_in_info=True,
        **kwargs,
    )

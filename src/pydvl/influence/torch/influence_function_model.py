"""
This module implements several implementations of [InfluenceFunctionModel]
[pydvl.influence.base_influence_function_model.InfluenceFunctionModel]
utilizing PyTorch.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pydvl.utils.progress import log_duration

from ..base_influence_function_model import (
    InfluenceFunctionModel,
    InfluenceMode,
    UnsupportedInfluenceModeException,
)
from .functional import (
    LowRankProductRepresentation,
    create_batch_hvp_function,
    create_hvp_function,
    create_matrix_jacobian_product_function,
    create_per_sample_gradient_function,
    create_per_sample_mixed_derivative_function,
    hessian,
    model_hessian_low_rank,
)
from .util import flatten_dimensions

logger = logging.getLogger(__name__)


class TorchInfluenceFunctionModel(
    InfluenceFunctionModel[torch.Tensor, DataLoader], ABC
):
    """
    Abstract base class for influence computation related to torch models
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        self.loss = loss
        self.model = model
        self._n_parameters = sum(
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
    def n_parameters(self):
        return self._n_parameters

    @property
    def is_thread_safe(self) -> bool:
        return False

    @property
    def model_device(self):
        return self._model_device

    @property
    def model_params(self):
        return self._model_params

    @log_duration
    def _loss_grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        grads = create_per_sample_gradient_function(self.model, self.loss)(
            self.model_params, x, y
        )
        shape = (x.shape[0], -1)
        return flatten_dimensions(grads.values(), shape=shape)

    @log_duration
    def _flat_loss_mixed_grad(self, x: torch.Tensor, y: torch.Tensor):
        mixed_grads = create_per_sample_mixed_derivative_function(
            self.model, self.loss
        )(self.model_params, x, y)
        shape = (*x.shape, -1)
        return flatten_dimensions(mixed_grads.values(), shape=shape)

    def influences(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> torch.Tensor:
        r"""
        Compute the approximation of

        \[
        \langle H^{-1}\nabla_{theta} \ell(y_{\text{test}}, f_{\theta}(x_{\text{test}})),
            \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle
        \]

        for the case of up-weighting influence, resp.

        \[
        \langle H^{-1}\nabla_{theta} \ell(y_{\text{test}}, f_{\theta}(x_{\text{test}})),
            \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle
        \]

        for the perturbation type influence case.

        Args:
            x_test: model input to use in the gradient computations
                of $H^{-1}\nabla_{theta} \ell(y_{\text{test}},
                    f_{\theta}(x_{\text{test}}))$
            y_test: label tensor to compute gradients
            x: optional model input to use in the gradient computations
                $\nabla_{theta}\ell(y, f_{\theta}(x))$,
                resp. $\nabla_{x}\nabla_{theta}\ell(y, f_{\theta}(x))$,
                if None, use $x=x_{\text{test}}$
            y: optional label tensor to compute gradients
            mode: enum value of [InfluenceType]
                [pydvl.influence.base_influence_model.InfluenceType]

        Returns:
            Tensor representing the element-wise scalar products for the provided batch

        """
        t: torch.Tensor = super().influences(x_test, y_test, x, y, mode=mode)
        return t

    def _influences(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> torch.Tensor:

        if not self.is_fitted:
            raise ValueError(
                "Instance must be fitted before calling influence methods on it"
            )

        if x is None:

            if y is not None:
                raise ValueError(
                    "Providing labels y, without providing model input x "
                    "is not supported"
                )

            return self._symmetric_values(
                x_test.to(self.model_device),
                y_test.to(self.model_device),
                mode,
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
            mode,
        )

    def _non_symmetric_values(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        mode: InfluenceMode = InfluenceMode.Up,
    ):
        if mode == InfluenceMode.Up:
            if x_test.shape[0] <= x.shape[0]:
                factor = self.influence_factors(x_test, y_test)
                values = self.influences_from_factors(factor, x, y, mode=mode)
            else:
                factor = self.influence_factors(x, y)
                values = self.influences_from_factors(
                    factor, x_test, y_test, mode=mode
                ).T
        elif mode == InfluenceMode.Perturbation:
            factor = self.influence_factors(x_test, y_test)
            values = self.influences_from_factors(factor, x, y, mode=mode)
        else:
            raise UnsupportedInfluenceModeException(mode)
        return values

    def _symmetric_values(
        self, x: torch.Tensor, y: torch.Tensor, mode: InfluenceMode
    ) -> torch.Tensor:

        grad = self._loss_grad(x, y)
        fac = self._solve_hvp(grad)

        if mode == InfluenceMode.Up:
            values = fac @ grad.T
        elif mode == InfluenceMode.Perturbation:
            values = self.influences_from_factors(fac, x, y, mode=mode)
        else:
            raise UnsupportedInfluenceModeException(mode)
        return values

    def influence_factors(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        Compute approximation of

        \[ H^{-1}\nabla_{\theta} \ell(y, f_{\theta}(x)) \]

        where the gradient is meant to be per sample of the batch $(x, y)$.

        Args:
            x: model input to use in the gradient computations
            y: label tensor to compute gradients

        Returns:
            Tensor representing the element-wise inverse Hessian matrix vector products

        """
        return super().influence_factors(x, y)

    def _influence_factors(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        if not self.is_fitted:
            raise ValueError(
                "Instance must be fitted before calling influence methods on it"
            )

        return self._solve_hvp(
            self._loss_grad(x.to(self.model_device), y.to(self.model_device))
        )

    def influences_from_factors(
        self,
        z_test_factors: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> torch.Tensor:
        r"""
        Computation of

        \[ \langle z_{\text{test_factors}},
            \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the case of up-weighting influence, resp.

        \[ \langle z_{\text{test_factors}},
            \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the perturbation type influence case. The gradient is meant to be per sample
        of the batch $(x, y)$.

        Args:
             z_test_factors: pre-computed tensor, approximating
                $H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
                    f_{\theta}(x_{\text{test}}))$
             x: model input to use in the gradient computations
                $\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                resp. $\nabla_{x}\nabla_{\theta}\ell(y, f_{\theta}(x))$
             y: label tensor to compute gradients
             mode: enum value of [InfluenceType]
                [pydvl.influence.twice_differentiable.InfluenceType]

        Returns:
            Tensor representing the element-wise scalar products for the provided batch

        """
        if mode == InfluenceMode.Up:
            return (
                z_test_factors
                @ self._loss_grad(x.to(self.model_device), y.to(self.model_device)).T
            )
        elif mode == InfluenceMode.Perturbation:
            return torch.einsum(
                "ia,j...a->ij...",
                z_test_factors,
                self._flat_loss_mixed_grad(
                    x.to(self.model_device), y.to(self.model_device)
                ),
            )
        else:
            raise UnsupportedInfluenceModeException(mode)

    @abstractmethod
    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        pass


class DirectInfluence(TorchInfluenceFunctionModel):
    r"""
    Given a model and training data, it finds x such that \(Hx = b\),
    with \(H\) being the model hessian.

    Args:
        model: instance of [torch.nn.Module][torch.nn.Module].
        hessian_regularization: Regularization of the hessian.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        hessian_regularization: float = 0.0,
    ):
        super().__init__(model, loss)
        self.hessian_regularization = hessian_regularization

    hessian: torch.Tensor

    @property
    def is_fitted(self):
        try:
            return self.hessian is not None
        except AttributeError:
            return False

    def fit(self, data: DataLoader) -> DirectInfluence:
        """
        Compute the hessian matrix based on a provided dataloader

        Args:
            data: Instance of [torch.utils.data.Dataloader]
                [torch.utils.data.Dataloader]

        Returns:
            The fitted instance
        """
        self.hessian = hessian(self.model, self.loss, data)
        return self

    @log_duration
    def influences(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> torch.Tensor:
        r"""
        Compute approximation of

        \[ \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
            f_{\theta}(x_{\text{test}})),
            \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle, \]

        for the case of up-weighting influence, resp.

        \[ \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
            f_{\theta}(x_{\text{test}})),
            \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the perturbation type influence case. The action of $H^{-1}$ is achieved
        via a direct solver using [torch.linalg.solve][torch.linalg.solve].

        Args:
            x_test: model input to use in the gradient computations of
                $H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
                    f_{\theta}(x_{\text{test}}))$
            y_test: label tensor to compute gradients
            x: optional model input to use in the gradient computations
                $\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                resp. $\nabla_{x}\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                if None, use $x=x_{\text{test}}$
            y: optional label tensor to compute gradients
            mode: enum value of [InfluenceType]
                [pydvl.influence.base_influence_model.InfluenceType]

        Returns:
            [torch.nn.Tensor][torch.nn.Tensor] representing the element-wise
                scalar products for the provided batch.

        """
        return super().influences(x_test, y_test, x, y, mode=mode)

    @log_duration
    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(
            self.hessian.to(self.model_device)
            + self.hessian_regularization
            * torch.eye(self.n_parameters, device=self.model_device),
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


class CgInfluence(TorchInfluenceFunctionModel):
    r"""
    Given a model and training data, it uses conjugate gradient to calculate the
    inverse of the Hessian Vector Product. More precisely, it finds x such that \(Hx =
    b\), with \(H\) being the model hessian. For more info, see
    [Conjugate Gradient][conjugate-gradient].

    Args:
        model: Instance of [torch.nn.Module][torch.nn.Module].
        loss: A callable that takes the model's output and target as input and returns
              the scalar loss.
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
        hessian_regularization: float = 0.0,
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

    train_dataloader: DataLoader

    @property
    def is_fitted(self):
        try:
            return self.train_dataloader is not None
        except AttributeError:
            return False

    def fit(self, data: DataLoader) -> CgInfluence:
        self.train_dataloader = data
        return self

    @log_duration
    def influences(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> torch.Tensor:
        r"""
        Compute approximation of

        \[ \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
            f_{\theta}(x_{\text{test}})),
            \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle, \]

        for the case of up-weighting influence, resp.

        \[ \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
            f_{\theta}(x_{\text{test}})),
            \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the perturbation type influence case. The approximate action of $H^{-1}$
        is achieved via the [conjugate gradient method]
        (https://en.wikipedia.org/wiki/Conjugate_gradient_method).

        Args:
            x_test: model input to use in the gradient computations of
                $H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
                    f_{\theta}(x_{\text{test}}))$
            y_test: label tensor to compute gradients
            x: optional model input to use in the gradient computations
                $\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                resp. $\nabla_{x}\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                if None, use $x=x_{\text{test}}$
            y: optional label tensor to compute gradients
            mode: enum value of [InfluenceType]
                [pydvl.influence.base_influence_model.InfluenceType]

        Returns:
            [torch.nn.Tensor][torch.nn.Tensor] representing the element-wise
                scalar products for the provided batch.

        """
        return super().influences(x_test, y_test, x, y, mode=mode)

    @log_duration
    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        if len(self.train_dataloader) == 0:
            raise ValueError("Training dataloader must not be empty.")

        hvp = create_hvp_function(self.model, self.loss, self.train_dataloader)

        def reg_hvp(v: torch.Tensor):
            return hvp(v) + self.hessian_regularization * v.type(rhs.dtype)

        batch_cg = torch.zeros_like(rhs)

        for idx, bi in enumerate(
            tqdm(rhs, disable=not self.progress, desc="Conjugate gradient")
        ):
            batch_result = self._solve_cg(
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

    @staticmethod
    def _solve_cg(
        hvp: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        *,
        x0: Optional[torch.Tensor] = None,
        rtol: float = 1e-7,
        atol: float = 1e-7,
        maxiter: Optional[int] = None,
    ) -> torch.Tensor:
        r"""
        Conjugate gradient solver for the Hessian vector product.

        Args:
            hvp: A callable Hvp, operating with tensors of size N.
            b: A vector or matrix, the right hand side of the equation \(Hx = b\).
            x0: Initial guess for hvp.
            rtol: Maximum relative tolerance of result.
            atol: Absolute tolerance of result.
            maxiter: Maximum number of iterations. If None, defaults to 10*len(b).

        Returns:
            [torch.nn.Tensor][torch.nn.Tensor] representing the solution of \(Ax=b\).
        """

        if x0 is None:
            x0 = torch.clone(b)
        if maxiter is None:
            maxiter = len(b) * 10

        y_norm = torch.sum(torch.matmul(b, b)).item()
        stopping_val = max([rtol**2 * y_norm, atol**2])

        x = x0
        p = r = (b - hvp(x)).squeeze()
        gamma = torch.sum(torch.matmul(r, r)).item()

        for k in range(maxiter):
            if gamma < stopping_val:
                break
            Ap = hvp(p).squeeze()
            alpha = gamma / torch.sum(torch.matmul(p, Ap)).item()
            x += alpha * p
            r -= alpha * Ap
            gamma_ = torch.sum(torch.matmul(r, r)).item()
            beta = gamma_ / gamma
            gamma = gamma_
            p = r + beta * p

        return x


class LissaInfluence(TorchInfluenceFunctionModel):
    r"""
    Uses LISSA, Linear time Stochastic Second-Order Algorithm, to iteratively
    approximate the inverse Hessian. More precisely, it finds x s.t. \(Hx = b\),
    with \(H\) being the model's second derivative wrt. the parameters.
    This is done with the update

    \[H^{-1}_{j+1} b = b + (I - d) \ H - \frac{H^{-1}_j b}{s},\]

    where \(I\) is the identity matrix, \(d\) is a dampening term and \(s\) a scaling
    factor that are applied to help convergence. For details,
    see [Linear time Stochastic Second-Order Approximation (LiSSA)]
    [linear-time-stochastic-second-order-approximation-lissa]

    Args:
        model: instance of [torch.nn.Module][torch.nn.Module].
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
        hessian_regularization: float = 0.0,
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

    train_dataloader: DataLoader

    @property
    def is_fitted(self):
        try:
            return self.train_dataloader is not None
        except AttributeError:
            return False

    def fit(self, data: DataLoader) -> LissaInfluence:
        self.train_dataloader = data
        return self

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
            create_batch_hvp_function(self.model, self.loss),
            in_dims=(None, None, None, 0),
        )
        for _ in tqdm(range(self.maxiter), disable=not self.progress, desc="Lissa"):
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


class ArnoldiInfluence(TorchInfluenceFunctionModel):
    r"""
    Solves the linear system Hx = b, where H is the Hessian of the model's loss function
    and b is the given right-hand side vector.
    It employs the [implicitly restarted Arnoldi method]
    (https://en.wikipedia.org/wiki/Arnoldi_iteration) for
    computing a partial eigen decomposition, which is used fo the inversion i.e.

    \[x = V D^{-1} V^T b\]

    where \(D\) is a diagonal matrix with the top (in absolute value) `rank_estimate`
    eigenvalues of the Hessian
    and \(V\) contains the corresponding eigenvectors.
    For more information, see [Arnoldi][arnoldi].

    Args:
        model: Instance of [torch.nn.Module][torch.nn.Module].
            The Hessian will be calculated with respect to this model's parameters.
        hessian_regularization: Optional regularization parameter added
            to the Hessian-vector product for numerical stability.
        rank_estimate: The number of eigenvalues and corresponding eigenvectors
            to compute. Represents the desired rank of the Hessian approximation.
        krylov_dimension: The number of Krylov vectors to use for the Lanczos method.
            Defaults to min(model's number of parameters,
            max(2 times rank_estimate + 1, 20)).
        tol: The stopping criteria for the Lanczos algorithm.
            Ignored if `low_rank_representation` is provided.
        max_iter: The maximum number of iterations for the Lanczos method.
            Ignored if `low_rank_representation` is provided.
        eigen_computation_on_gpu: If True, tries to execute the eigen pair approximation
            on the model's device
            via a cupy implementation. Ensure the model size or rank_estimate
            is appropriate for device memory.
            If False, the eigen pair approximation is executed on the CPU by the scipy
            wrapper to ARPACK.
    """
    low_rank_representation: LowRankProductRepresentation

    def __init__(
        self,
        model,
        loss,
        hessian_regularization: float = 0.0,
        rank_estimate: int = 10,
        krylov_dimension: Optional[int] = None,
        tol: float = 1e-6,
        max_iter: Optional[int] = None,
        eigen_computation_on_gpu: bool = False,
    ):

        super().__init__(model, loss)
        self.hessian_regularization = hessian_regularization
        self.rank_estimate = rank_estimate
        self.tol = tol
        self.max_iter = max_iter
        self.krylov_dimension = krylov_dimension
        self.eigen_computation_on_gpu = eigen_computation_on_gpu

    @property
    def is_fitted(self):
        try:
            return self.low_rank_representation is not None
        except AttributeError:
            return False

    def fit(self, data: DataLoader) -> ArnoldiInfluence:
        r"""
        Fitting corresponds to the computation of the low rank decomposition

        \[ V D^{-1} V^T \]

        of the Hessian defined by the provided data loader.

        Args:
            data: Instance of [torch.utils.data.Dataloader][torch.utils.data.Dataloader]

        Returns:
            The fitted instance

        """
        low_rank_representation = model_hessian_low_rank(
            self.model,
            self.loss,
            data,
            hessian_perturbation=0.0,  # regularization is applied, when computing values
            rank_estimate=self.rank_estimate,
            krylov_dimension=self.krylov_dimension,
            tol=self.tol,
            max_iter=self.max_iter,
            eigen_computation_on_gpu=self.eigen_computation_on_gpu,
        )
        self.low_rank_representation = low_rank_representation.to(self.model_device)
        return self

    def _non_symmetric_values(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> torch.Tensor:

        if mode == InfluenceMode.Up:
            mjp = create_matrix_jacobian_product_function(
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
        elif mode == InfluenceMode.Perturbation:
            factors = self.influence_factors(x_test, y_test)
            values = self.influences_from_factors(factors, x, y, mode=mode)
        else:
            raise UnsupportedInfluenceModeException(mode)
        return values

    def _symmetric_values(
        self, x: torch.Tensor, y: torch.Tensor, mode: InfluenceMode
    ) -> torch.Tensor:

        if mode == InfluenceMode.Up:
            left = create_matrix_jacobian_product_function(
                self.model, self.loss, self.low_rank_representation.projections.T
            )(self.model_params, x, y)
            regularized_eigenvalues = (
                self.low_rank_representation.eigen_vals + self.hessian_regularization
            )
            right = torch.diag_embed(1.0 / regularized_eigenvalues) @ left
            values = torch.einsum("ij, ik -> jk", left, right)
        elif mode == InfluenceMode.Perturbation:
            factors = self.influence_factors(x, y)
            values = self.influences_from_factors(factors, x, y, mode=mode)
        else:
            raise UnsupportedInfluenceModeException(mode)
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

"""
This module implements several implementations of [InfluenceFunctionModel]
[pydvl.influence.base_influence_function_model.InfluenceFunctionModel]
utilizing PyTorch.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ...utils.progress import log_duration
from .. import InfluenceMode
from ..base_influence_function_model import (
    ComposableInfluence,
    InfluenceFunctionModel,
    NotImplementedLayerRepresentationException,
    UnsupportedInfluenceModeException,
)
from ..types import BlockMapper, OperatorGradientComposition
from .functional import (
    LowRankProductRepresentation,
    create_batch_hvp_function,
    create_hvp_function,
    create_matrix_jacobian_product_function,
    create_per_sample_gradient_function,
    create_per_sample_mixed_derivative_function,
    hessian,
    model_hessian_low_rank,
    model_hessian_nystroem_approximation,
)
from .operator.base import TorchOperator
from .operator.gradient_provider import (
    TorchPerSampleAutoGrad,
    TorchPerSampleGradientProvider,
)
from .operator.solve import InverseHarmonicMeanOperator
from .pre_conditioner import PreConditioner
from .util import (
    BlockMode,
    EkfacRepresentation,
    LossType,
    ModelInfoMixin,
    ModelParameterDictBuilder,
    TorchBatch,
    empirical_cross_entropy_loss_fn,
    flatten_dimensions,
    safe_torch_linalg_eigh,
)

__all__ = [
    "DirectInfluence",
    "CgInfluence",
    "LissaInfluence",
    "ArnoldiInfluence",
    "EkfacInfluence",
    "NystroemSketchInfluence",
]

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
        self._model_dtype = next(
            (p.dtype for p in model.parameters() if p.requires_grad)
        )
        super().__init__()

    @property
    def model_dtype(self):
        return self._model_dtype

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
    def _flat_loss_mixed_grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
        \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
        f_{\theta}(x_{\text{test}})), \nabla_{\theta} \ell(y, f_{\theta}(x))\rangle
        \]

        for the case of up-weighting influence, resp.

        \[
        \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}}, f_{\theta}(x_{\text{test}})),
            \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle
        \]

        for the perturbation type influence case. For all input tensors it is assumed,
        that the first dimension is the batch dimension (in case, you want to provide
        a single sample z, call z.unsqueeze(0) if no batch dimension is present).

        Args:
            x_test: model input to use in the gradient computations
                of $H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
                    f_{\theta}(x_{\text{test}}))$
            y_test: label tensor to compute gradients
            x: optional model input to use in the gradient computations
                $\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                resp. $\nabla_{x}\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                if None, use $x=x_{\text{test}}$
            y: optional label tensor to compute gradients
            mode: enum value of [InfluenceMode]
                [pydvl.influence.base_influence_function_model.InfluenceMode]

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
    ) -> torch.Tensor:
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
        For all input tensors it is assumed,
        that the first dimension is the batch dimension (in case, you want to provide
        a single sample z, call z.unsqueeze(0) if no batch dimension is present).

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
        of the batch $(x, y)$. For all input tensors it is assumed,
        that the first dimension is the batch dimension (in case, you want to provide
        a single sample z, call z.unsqueeze(0) if no batch dimension is present).

        Args:
            z_test_factors: pre-computed tensor, approximating
                $H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
                f_{\theta}(x_{\text{test}}))$
            x: model input to use in the gradient computations
                $\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                resp. $\nabla_{x}\nabla_{\theta}\ell(y, f_{\theta}(x))$
            y: label tensor to compute gradients
            mode: enum value of [InfluenceMode]
                [pydvl.influence.base_influence_function_model.InfluenceMode]

        Returns:
            Tensor representing the element-wise scalar products for the provided batch

        """
        if mode == InfluenceMode.Up:
            return (
                z_test_factors.to(self.model_device)
                @ self._loss_grad(x.to(self.model_device), y.to(self.model_device)).T
            )
        elif mode == InfluenceMode.Perturbation:
            return torch.einsum(
                "ia,j...a->ij...",
                z_test_factors.to(self.model_device),
                self._flat_loss_mixed_grad(
                    x.to(self.model_device), y.to(self.model_device)
                ),
            )
        else:
            raise UnsupportedInfluenceModeException(mode)

    @abstractmethod
    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        pass

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        self._model_params = {
            k: p.detach().to(device)
            for k, p in self.model.named_parameters()
            if p.requires_grad
        }
        self._model_device = device
        return self


class DirectInfluence(TorchInfluenceFunctionModel):
    r"""
    Given a model and training data, it finds x such that \(Hx = b\),
    with \(H\) being the model hessian.

    Args:
        model: A PyTorch model. The Hessian will be calculated with respect to
            this model's parameters.
        loss: A callable that takes the model's output and target as input and returns
              the scalar loss.
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

    @log_duration(log_level=logging.INFO)
    def fit(self, data: DataLoader) -> DirectInfluence:
        """
        Compute the hessian matrix based on a provided dataloader.

        Args:
            data: The data to compute the Hessian with.

        Returns:
            The fitted instance.
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
            mode: enum value of [InfluenceMode]
                [pydvl.influence.base_influence_function_model.InfluenceMode]

        Returns:
            A tensor representing the element-wise scalar products for the
                provided batch.

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
        if self.is_fitted:
            self.hessian = self.hessian.to(device)
        return super().to(device)


class CgInfluence(TorchInfluenceFunctionModel):
    r"""
    Given a model and training data, it uses conjugate gradient to calculate the
    inverse of the Hessian Vector Product. More precisely, it finds x such that \(Hx =
    b\), with \(H\) being the model hessian. For more info, see
    [Conjugate Gradient][conjugate-gradient].

    Args:
        model: A PyTorch model. The Hessian will be calculated with respect to
            this model's parameters.
        loss: A callable that takes the model's output and target as input and returns
              the scalar loss.
        hessian_regularization: Optional regularization parameter added
            to the Hessian-vector product for numerical stability.
        x0: Initial guess for hvp. If None, defaults to b.
        rtol: Maximum relative tolerance of result.
        atol: Absolute tolerance of result.
        maxiter: Maximum number of iterations. If None, defaults to 10*len(b).
        progress: If True, display progress bars for computing in the non-block mode
            (use_block_cg=False).
        precompute_grad: If True, the full data gradient is precomputed and kept
            in memory, which can speed up the hessian vector product computation.
            Set this to False, if you can't afford to keep the full computation graph
            in memory.
        pre_conditioner: Optional pre-conditioner to improve convergence of conjugate
            gradient method
        use_block_cg: If True, use block variant of conjugate gradient method, which
            solves several right hand sides simultaneously
        warn_on_max_iteration: If True, logs a warning, if the desired tolerance is not
            achieved within `maxiter` iterations. If False, the log level for this
            information is `logging.DEBUG`

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
        precompute_grad: bool = False,
        pre_conditioner: Optional[PreConditioner] = None,
        use_block_cg: bool = False,
        warn_on_max_iteration: bool = True,
    ):
        super().__init__(model, loss)
        self.warn_on_max_iteration = warn_on_max_iteration
        self.use_block_cg = use_block_cg
        self.pre_conditioner = pre_conditioner
        self.precompute_grad = precompute_grad
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

    @log_duration(log_level=logging.INFO)
    def fit(self, data: DataLoader) -> CgInfluence:
        self.train_dataloader = data
        if self.pre_conditioner is not None:
            hvp = create_hvp_function(
                self.model,
                self.loss,
                self.train_dataloader,
                precompute_grad=self.precompute_grad,
            )

            def model_hessian_mat_mat_prod(x: torch.Tensor):
                return torch.func.vmap(hvp, in_dims=1, randomness="same")(x).t()

            self.pre_conditioner.fit(
                model_hessian_mat_mat_prod,
                self.n_parameters,
                self.model_dtype,
                self.model_device,
                self.hessian_regularization,
            )
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
        Compute an approximation of

        \[ \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
            f_{\theta}(x_{\text{test}})),
            \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle, \]

        for the case of up-weighting influence, resp.

        \[ \langle H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
            f_{\theta}(x_{\text{test}})),
            \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the case of perturbation-type influence. The approximate action of
        $H^{-1}$ is achieved via the [conjugate gradient
        method](https://en.wikipedia.org/wiki/Conjugate_gradient_method).

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
            mode: enum value of [InfluenceMode]
                [pydvl.influence.base_influence_function_model.InfluenceMode]

        Returns:
            A tensor representing the element-wise scalar products for the
                provided batch.

        """
        return super().influences(x_test, y_test, x, y, mode=mode)

    @log_duration
    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        if len(self.train_dataloader) == 0:
            raise ValueError("Training dataloader must not be empty.")

        if self.use_block_cg:
            return self._solve_pbcg(rhs)

        hvp = create_hvp_function(
            self.model,
            self.loss,
            self.train_dataloader,
            precompute_grad=self.precompute_grad,
        )

        def reg_hvp(v: torch.Tensor):
            return hvp(v) + self.hessian_regularization * v.type(rhs.dtype)

        y_norm = torch.linalg.norm(rhs, dim=0)

        stopping_val = torch.clamp(self.rtol**2 * y_norm, min=self.atol**2)

        batch_cg = torch.zeros_like(rhs)

        for idx, (bi, _tol) in enumerate(
            tqdm(
                zip(rhs, stopping_val),
                disable=not self.progress,
                desc="Conjugate gradient",
            )
        ):
            batch_result = self._solve_pcg(
                reg_hvp,
                bi,
                tol=_tol,
                x0=self.x0,
                maxiter=self.maxiter,
            )
            batch_cg[idx] = batch_result

        return batch_cg

    def _solve_pcg(
        self,
        hvp: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        *,
        tol: float,
        x0: Optional[torch.Tensor] = None,
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
            A tensor with the solution of \(Ax=b\).
        """

        if x0 is None:
            x0 = torch.clone(b)
        if maxiter is None:
            maxiter = len(b) * 10

        x = x0

        r0 = b - hvp(x)

        if self.pre_conditioner is not None:
            p = z0 = self.pre_conditioner.solve(r0)
        else:
            p = z0 = r0

        for k in range(maxiter):
            if torch.norm(r0) < tol:
                logger.debug(f"Terminated cg after {k} iterations with residuum={r0}")
                break
            Ap = hvp(p)
            alpha = torch.dot(r0, z0) / torch.dot(p, Ap)
            x += alpha * p
            r = r0 - alpha * Ap

            if self.pre_conditioner is not None:
                z = self.pre_conditioner.solve(r)
            else:
                z = r

            beta = torch.dot(r, z) / torch.dot(r0, z0)

            r0 = r
            p = z + beta * p
            z0 = z
        else:
            log_level = logging.WARNING if self.warn_on_max_iteration else logging.DEBUG
            logger.log(
                log_level,
                f"Reached max number of iterations {maxiter=} without "
                f"achieving the desired tolerance {tol}. \n"
                f"Achieved residuum is {torch.norm(r0)}.\n"
                f"Consider increasing 'maxiter', the desired tolerance or the "
                f"parameter 'hessian_regularization'.",
            )

        return x

    def _solve_pbcg(
        self,
        rhs: torch.Tensor,
    ):
        hvp = create_hvp_function(
            self.model,
            self.loss,
            self.train_dataloader,
            precompute_grad=self.precompute_grad,
        )

        # The block variant of conjugate gradient is known to suffer from breakdown,
        # due to the possibility of rank deficiency of the iterates of the parameter
        # matrix P^tAP, which destabilizes the direct solver.
        # The paper `Randomized Nyström Preconditioning,
        # Frangella, Zachary and Tropp, Joel A. and Udell, Madeleine,
        # SIAM J. Matrix Anal. Appl., 2023`
        # proposes a simple orthogonalization pre-processing. However, we observed, that
        # this stabilization only worked for double precision. We thus implement
        # a different stabilization strategy described in
        # `A breakdown-free block conjugate gradient method, Ji, Hao and Li, Yaohang,
        # BIT Numerical Mathematics, 2017`

        def mat_mat(x: torch.Tensor):
            return torch.vmap(
                lambda u: hvp(u) + self.hessian_regularization * u,
                in_dims=1,
                randomness="same",
            )(x)

        X = torch.clone(rhs.T)

        R = (rhs - mat_mat(X)).T
        Z = R if self.pre_conditioner is None else self.pre_conditioner.solve(R)
        P, _, _ = torch.linalg.svd(Z, full_matrices=False)
        active_indices = torch.as_tensor(
            list(range(X.shape[-1])), dtype=torch.long, device=self.model_device
        )

        maxiter = self.maxiter if self.maxiter is not None else len(rhs) * 10
        y_norm = torch.linalg.norm(rhs, dim=1)
        tol = torch.clamp(self.rtol**2 * y_norm, min=self.atol**2)

        # In the case the parameter dimension is smaller than the number of right
        # hand sides, we do not shrink the indices due to resulting wrong
        # dimensionality of the svd decomposition. We consider this an edge case, which
        # does not need optimization
        shrink_finished_indices = rhs.shape[0] <= rhs.shape[1]

        for k in range(maxiter):
            Q = mat_mat(P).T
            p_t_ap = P.T @ Q
            alpha = torch.linalg.solve(p_t_ap, P.T @ R)
            X[:, active_indices] += P @ alpha
            R -= Q @ alpha

            B = torch.linalg.norm(R, dim=0)
            non_finished_indices = torch.nonzero(B > tol)
            num_remaining_indices = non_finished_indices.numel()
            non_finished_indices = non_finished_indices.squeeze()

            if num_remaining_indices == 1:
                non_finished_indices = non_finished_indices.unsqueeze(-1)

            if num_remaining_indices == 0:
                logger.debug(
                    f"Terminated block cg after {k} iterations with max "
                    f"residuum={B.max()}"
                )
                break

            # Reduce problem size by removing finished columns from the iteration
            if shrink_finished_indices:
                active_indices = active_indices[non_finished_indices]
                R = R[:, non_finished_indices]
                P = P[:, non_finished_indices]
                Q = Q[:, non_finished_indices]
                p_t_ap = p_t_ap[:, non_finished_indices][non_finished_indices, :]
                tol = tol[non_finished_indices]

            Z = R if self.pre_conditioner is None else self.pre_conditioner.solve(R)
            beta = -torch.linalg.solve(p_t_ap, Q.T @ Z)
            Z_tmp = Z + P @ beta

            if Z_tmp.ndim == 1:
                Z_tmp = Z_tmp.unsqueeze(-1)

            # Orthogonalization search directions to stabilize the action of
            # (P^tAP)^{-1}
            P, _, _ = torch.linalg.svd(Z_tmp, full_matrices=False)
        else:
            log_level = logging.WARNING if self.warn_on_max_iteration else logging.DEBUG
            logger.log(
                log_level,
                f"Reached max number of iterations {maxiter=} of block cg "
                f"without achieving the desired tolerance {tol.min()}. \n"
                f"Achieved max residuum is "
                f"{torch.linalg.norm(R, dim=0).max()}.\n"
                f"Consider increasing 'maxiter', the desired tolerance or "
                f"the parameter 'hessian_regularization'.",
            )

        return X.T

    def to(self, device: torch.device):
        if self.pre_conditioner is not None:
            self.pre_conditioner = self.pre_conditioner.to(device)
        return super().to(device)


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
        model: A PyTorch model. The Hessian will be calculated with respect to
            this model's parameters.
        loss: A callable that takes the model's output and target as input and returns
              the scalar loss.
        hessian_regularization: Optional regularization parameter added
            to the Hessian-vector product for numerical stability.
        maxiter: Maximum number of iterations.
        dampen: Dampening factor, defaults to 0 for no dampening.
        scale: Scaling factor, defaults to 10.
        h0: Initial guess for hvp.
        rtol: tolerance to use for early stopping
        progress: If True, display progress bars.
        warn_on_max_iteration: If True, logs a warning, if the desired tolerance is not
            achieved within `maxiter` iterations. If False, the log level for this
            information is `logging.DEBUG`
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
        warn_on_max_iteration: bool = True,
    ):
        super().__init__(model, loss)
        self.warn_on_max_iteration = warn_on_max_iteration
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

    @log_duration(log_level=logging.INFO)
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
        for k in tqdm(
            range(self.maxiter), disable=not self.progress, desc="Lissa iteration"
        ):
            x, y = next(iter(shuffled_training_data))
            x = x.to(self.model_device)
            y = y.to(self.model_device)
            reg_hvp = (
                lambda v: b_hvp(model_params, x, y, v) + self.hessian_regularization * v
            )
            residual = lissa_step(h_estimate, reg_hvp) - h_estimate
            h_estimate += residual
            if torch.isnan(h_estimate).any():
                raise RuntimeError("NaNs in h_estimate. Increase scale or dampening.")
            max_residual = torch.max(torch.abs(residual / h_estimate))
            if max_residual < self.rtol:
                mean_residual = torch.mean(torch.abs(residual / h_estimate))
                logger.debug(
                    f"Terminated Lissa after {k} iterations with "
                    f"{max_residual*100:.2f} % max residual and"
                    f" mean residual {mean_residual*100:.5f} %"
                )
                break
        else:
            mean_residual = torch.mean(torch.abs(residual / h_estimate))
            log_level = logging.WARNING if self.warn_on_max_iteration else logging.DEBUG
            logger.log(
                log_level,
                f"Reached max number of iterations {self.maxiter} without "
                f"achieving the desired tolerance {self.rtol}.\n "
                f"Achieved max residual {max_residual*100:.2f} % and"
                f" {mean_residual*100:.5f} % mean residual",
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
        model: A PyTorch model. The Hessian will be calculated with respect to
            this model's parameters.
        loss: A callable that takes the model's output and target as input and returns
              the scalar loss.
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
        precompute_grad: If True, the full data gradient is precomputed and kept
            in memory, which can speed up the hessian vector product computation.
            Set this to False, if you can't afford to keep the full computation graph
            in memory.
    """

    low_rank_representation: LowRankProductRepresentation

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        hessian_regularization: float = 0.0,
        rank_estimate: int = 10,
        krylov_dimension: Optional[int] = None,
        tol: float = 1e-6,
        max_iter: Optional[int] = None,
        eigen_computation_on_gpu: bool = False,
        precompute_grad: bool = False,
    ):
        super().__init__(model, loss)
        self.hessian_regularization = hessian_regularization
        self.rank_estimate = rank_estimate
        self.tol = tol
        self.max_iter = max_iter
        self.krylov_dimension = krylov_dimension
        self.eigen_computation_on_gpu = eigen_computation_on_gpu
        self.precompute_grad = precompute_grad

    @property
    def is_fitted(self):
        try:
            return self.low_rank_representation is not None
        except AttributeError:
            return False

    @log_duration(log_level=logging.INFO)
    def fit(self, data: DataLoader) -> ArnoldiInfluence:
        r"""
        Fitting corresponds to the computation of the low rank decomposition

        \[ V D^{-1} V^T \]

        of the Hessian defined by the provided data loader.

        Args:
            data: The data to compute the Hessian with.

        Returns:
            The fitted instance.

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
            precompute_grad=self.precompute_grad,
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

            inverse_regularized_eigenvalues = 1.0 / (
                self.low_rank_representation.eigen_vals + self.hessian_regularization
            )

            right = mjp(
                self.model_params, x, y
            ) * inverse_regularized_eigenvalues.unsqueeze(-1)
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
            inverse_regularized_eigenvalues = 1.0 / (
                self.low_rank_representation.eigen_vals + self.hessian_regularization
            )
            right = left * inverse_regularized_eigenvalues.unsqueeze(-1)
            values = torch.einsum("ij, ik -> jk", left, right)
        elif mode == InfluenceMode.Perturbation:
            factors = self.influence_factors(x, y)
            values = self.influences_from_factors(factors, x, y, mode=mode)
        else:
            raise UnsupportedInfluenceModeException(mode)
        return values

    @log_duration
    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        inverse_regularized_eigenvalues = 1.0 / (
            self.low_rank_representation.eigen_vals + self.hessian_regularization
        )

        projected_rhs = self.low_rank_representation.projections.t() @ rhs.t()
        result = self.low_rank_representation.projections @ (
            projected_rhs * inverse_regularized_eigenvalues.unsqueeze(-1)
        )

        return result.t()

    def to(self, device: torch.device):
        if self.is_fitted:
            self.low_rank_representation = self.low_rank_representation.to(device)
        return super().to(device)


class EkfacInfluence(TorchInfluenceFunctionModel):
    r"""
    Approximately solves the linear system Hx = b, where H is the Hessian of a model
    with the empirical categorical cross entropy as loss function and b is the given
    right-hand side vector.
    It employs the EK-FAC method, which is based on the kronecker
    factorization of the Hessian.

    Contrary to the other influence function methods, this implementation can only
    be used for classification tasks with a cross entropy loss function. However, it
    is much faster than the other methods and can be used efficiently for very large
    datasets and models. For more information,
    see [Eigenvalue Corrected K-FAC][eigenvalue-corrected-k-fac].

    Args:
        model: A PyTorch model. The Hessian will be calculated with respect to
            this model's parameters.
        update_diagonal: If True, the diagonal values in the ekfac representation are
            refitted from the training data after calculating the KFAC blocks.
            This provides a more accurate approximation of the Hessian, but it is
            computationally more expensive.
        hessian_regularization: Regularization of the hessian.
        progress: If True, display progress bars.
    """

    ekfac_representation: EkfacRepresentation

    def __init__(
        self,
        model: nn.Module,
        update_diagonal: bool = False,
        hessian_regularization: float = 0.0,
        progress: bool = False,
    ):
        super().__init__(model, torch.nn.functional.cross_entropy)
        self.hessian_regularization = hessian_regularization
        self.update_diagonal = update_diagonal
        self.active_layers = self._parse_active_layers()
        self.progress = progress

    @property
    def is_fitted(self):
        try:
            return self.ekfac_representation is not None
        except AttributeError:
            return False

    def _parse_active_layers(self) -> Dict[str, torch.nn.Module]:
        """
        Find all layers of the model that have parameters that require grad
        and return them in a dictionary. If a layer has some parameters that require
        grad and some that do not, raise an error.
        """
        active_layers: Dict[str, torch.nn.Module] = {}
        for m_name, module in self.model.named_modules():
            if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
                layer_requires_grad = [
                    param.requires_grad for param in module.parameters()
                ]
                if all(layer_requires_grad):
                    active_layers[m_name] = module
                elif any(layer_requires_grad):
                    raise ValueError(
                        f"Layer {m_name} has some parameters that require grad and some that do not."
                        f"This is not supported. Please set all parameters of the layer to require grad."
                    )
        return active_layers

    @staticmethod
    def _init_layer_kfac_blocks(
        module: torch.nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize the tensors that will store the cumulative forward and
        backward KFAC blocks for the layer.
        """
        if isinstance(module, nn.Linear):
            with_bias = module.bias is not None
            sG = module.out_features
            sA = module.in_features + int(with_bias)
            forward_x_layer = torch.zeros((sA, sA), device=module.weight.device)
            grad_y_layer = torch.zeros((sG, sG), device=module.weight.device)
        else:
            raise NotImplementedLayerRepresentationException(module_id=str(module))
        return forward_x_layer, grad_y_layer

    @staticmethod
    def _get_layer_kfac_hooks(
        m_name: str,
        module: torch.nn.Module,
        forward_x: Dict[str, torch.Tensor],
        grad_y: Dict[str, torch.Tensor],
    ) -> Tuple[Callable, Callable]:
        """
        Create the hooks that will be used to compute the forward and backward KFAC
        blocks for the layer. The hooks are registered to the layer and will be called
        during the forward and backward passes. At each pass, the hooks will update the
        tensors that store the cumulative forward and backward KFAC blocks for the layer.
        These tensors are stored in the forward_x and grad_y dictionaries.
        """
        if isinstance(module, nn.Linear):
            with_bias = module.bias is not None

            def input_hook(m, x, y):
                x = x[0].reshape(-1, module.in_features)
                if with_bias:
                    x = torch.cat(
                        (x, torch.ones((x.shape[0], 1), device=module.weight.device)),
                        dim=1,
                    )
                forward_x[m_name] += torch.mm(x.t(), x)

            def grad_hook(m, m_grad, m_out):
                m_out = m_out[0].reshape(-1, module.out_features)
                grad_y[m_name] += torch.mm(m_out.t(), m_out)

        else:
            raise NotImplementedLayerRepresentationException(module_id=str(module))
        return input_hook, grad_hook

    def _get_kfac_blocks(
        self,
        data: DataLoader,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Compute the KFAC blocks for each layer of the model, using the provided data.
        Returns the average forward and backward KFAC blocks for each layer in
        dictionaries.
        """
        forward_x = {}
        grad_y = {}
        hooks = []
        data_len = 0

        for m_name, module in self.active_layers.items():
            forward_x[m_name], grad_y[m_name] = self._init_layer_kfac_blocks(module)
            layer_input_hook, layer_grad_hook = self._get_layer_kfac_hooks(
                m_name, module, forward_x, grad_y
            )
            hooks.append(module.register_forward_hook(layer_input_hook))
            hooks.append(module.register_full_backward_hook(layer_grad_hook))

        for x, *_ in tqdm(
            data, disable=not self.progress, desc="K-FAC blocks - batch progress"
        ):
            data_len += x.shape[0]
            pred_y = self.model(x.to(self.model_device))
            loss = empirical_cross_entropy_loss_fn(pred_y)
            loss.backward()

        for key in forward_x.keys():
            forward_x[key] /= data_len
            grad_y[key] /= data_len

        for hook in hooks:
            hook.remove()

        return forward_x, grad_y

    @log_duration(log_level=logging.INFO)
    def fit(self, data: DataLoader) -> EkfacInfluence:
        """
        Compute the KFAC blocks for each layer of the model, using the provided data.
        It then creates an EkfacRepresentation object that stores the KFAC blocks for
        each layer, their eigenvalue decomposition and diagonal values.
        """
        forward_x, grad_y = self._get_kfac_blocks(data)
        layers_evecs_a = {}
        layers_evect_g = {}
        layers_diags = {}
        for key in self.active_layers.keys():
            evals_a, evecs_a = safe_torch_linalg_eigh(forward_x[key])
            evals_g, evecs_g = safe_torch_linalg_eigh(grad_y[key])
            layers_evecs_a[key] = evecs_a
            layers_evect_g[key] = evecs_g
            layers_diags[key] = torch.kron(evals_g.view(-1, 1), evals_a.view(-1, 1))

        self.ekfac_representation = EkfacRepresentation(
            self.active_layers.keys(),
            self.active_layers.values(),
            layers_evecs_a.values(),
            layers_evect_g.values(),
            layers_diags.values(),
        )
        if self.update_diagonal:
            self._update_diag(data)
        return self

    @staticmethod
    def _init_layer_diag(module: torch.nn.Module) -> torch.Tensor:
        """
        Initialize the tensor that will store the updated diagonal values of the layer.
        """
        if isinstance(module, nn.Linear):
            with_bias = module.bias is not None
            sG = module.out_features
            sA = module.in_features + int(with_bias)
            layer_diag = torch.zeros((sA * sG), device=module.weight.device)
        else:
            raise NotImplementedLayerRepresentationException(module_id=str(module))
        return layer_diag

    def _get_layer_diag_hooks(
        self,
        m_name: str,
        module: torch.nn.Module,
        last_x_kfe: Dict[str, torch.Tensor],
        diags: Dict[str, torch.Tensor],
    ) -> Tuple[Callable, Callable]:
        """
        Create the hooks that will be used to update the diagonal values of the layer.
        The hooks are registered to the layer and will be called during the forward and
        backward passes. At each pass, the hooks will update the tensor that stores the
        updated diagonal values of the layer. This tensor is stored in the diags
        dictionary.
        """
        evecs_a, evecs_g = self.ekfac_representation.get_layer_evecs()
        if isinstance(module, nn.Linear):
            with_bias = module.bias is not None

            def input_hook(m, x, y):
                x = x[0].reshape(-1, module.in_features)
                if with_bias:
                    x = torch.cat(
                        (x, torch.ones((x.shape[0], 1), device=module.weight.device)),
                        dim=1,
                    )
                last_x_kfe[m_name] = torch.mm(x, evecs_a[m_name])

            def grad_hook(m, m_grad, m_out):
                m_out = m_out[0].reshape(-1, module.out_features)
                gy_kfe = torch.mm(m_out, evecs_g[m_name])
                diags[m_name] += torch.mm(
                    gy_kfe.t() ** 2, last_x_kfe[m_name] ** 2
                ).view(-1)

        else:
            raise NotImplementedLayerRepresentationException(module_id=str(module))
        return input_hook, grad_hook

    def _update_diag(
        self,
        data: DataLoader,
    ) -> EkfacInfluence:
        """
        Compute the updated diagonal values for each layer of the model, using the
        provided data. It then updates the EkfacRepresentation object that stores the
        KFAC blocks for each layer, their eigenvalue decomposition and diagonal values.
        """
        if not self.is_fitted:
            raise ValueError(
                "EkfacInfluence must be fitted before updating the diagonal."
            )
        diags = {}
        last_x_kfe: Dict[str, torch.Tensor] = {}
        hooks = []
        data_len = 0

        for m_name, module in self.active_layers.items():
            diags[m_name] = self._init_layer_diag(module)
            input_hook, grad_hook = self._get_layer_diag_hooks(
                m_name, module, last_x_kfe, diags
            )
            hooks.append(module.register_forward_hook(input_hook))
            hooks.append(module.register_full_backward_hook(grad_hook))

        for x, *_ in tqdm(
            data, disable=not self.progress, desc="Update Diagonal - batch progress"
        ):
            data_len += x.shape[0]
            pred_y = self.model(x.to(self.model_device))
            loss = empirical_cross_entropy_loss_fn(pred_y)
            loss.backward()

        for key in diags.keys():
            diags[key] /= data_len

        for hook in hooks:
            hook.remove()

        self.ekfac_representation = EkfacRepresentation(
            self.ekfac_representation.layer_names,
            self.ekfac_representation.layers_module,
            self.ekfac_representation.evecs_a,
            self.ekfac_representation.evecs_g,
            diags.values(),
        )

        return self

    @staticmethod
    def _solve_hvp_by_layer(
        rhs: torch.Tensor,
        ekfac_representation: EkfacRepresentation,
        hessian_regularization: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the Hessian Vector Product for each layer of the model, using the
        provided ekfac representation and hessian regularization. It returns a
        dictionary containing the Hessian Vector Product for each layer.
        """
        hvp_layers = {}
        start_idx = 0
        for layer_id, (_, evecs_a, evecs_g, diag) in ekfac_representation:
            end_idx = start_idx + diag.shape[0]
            rhs_layer = rhs[:, start_idx : end_idx - evecs_g.shape[0]].reshape(
                rhs.shape[0], evecs_g.shape[0], -1
            )
            bias_layer_b = rhs[:, end_idx - evecs_g.shape[0] : end_idx]
            rhs_layer = torch.cat([rhs_layer, bias_layer_b.unsqueeze(2)], dim=2)
            v_kfe = torch.einsum(
                "bij,jk->bik",
                torch.einsum("ij,bjk->bik", evecs_g.t(), rhs_layer),
                evecs_a,
            )
            inv_diag = 1 / (diag.reshape(*v_kfe.shape[1:]) + hessian_regularization)
            inv_kfe = torch.einsum("bij,ij->bij", v_kfe, inv_diag)
            inv = torch.einsum(
                "bij,jk->bik",
                torch.einsum("ij,bjk->bik", evecs_g, inv_kfe),
                evecs_a.t(),
            )
            hvp_layers[layer_id] = torch.cat(
                [inv[:, :, :-1].reshape(rhs.shape[0], -1), inv[:, :, -1]], dim=1
            )
            start_idx = end_idx
        return hvp_layers

    @log_duration
    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        x = rhs.clone()
        start_idx = 0
        layer_hvp = self._solve_hvp_by_layer(
            rhs, self.ekfac_representation, self.hessian_regularization
        )
        for hvp in layer_hvp.values():
            end_idx = start_idx + hvp.shape[1]
            x[:, start_idx:end_idx] = hvp
            start_idx = end_idx
        x.detach_()
        return x

    def influences_by_layer(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Compute the influence of the data on the test data for each layer of the model.

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
            mode: enum value of [InfluenceMode]
                [pydvl.influence.base_influence_function_model.InfluenceMode]

        Returns:
            A dictionary containing the influence of the data on the test data for each
            layer of the model, with the layer name as key.
        """
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

            return self._symmetric_values_by_layer(
                x_test.to(self.model_device),
                y_test.to(self.model_device),
                mode,
            )

        if y is None:
            raise ValueError(
                "Providing model input x without providing labels y is not supported"
            )

        return self._non_symmetric_values_by_layer(
            x_test.to(self.model_device),
            y_test.to(self.model_device),
            x.to(self.model_device),
            y.to(self.model_device),
            mode,
        )

    def influence_factors_by_layer(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Computes the approximation of

        \[ H^{-1}\nabla_{\theta} \ell(y, f_{\theta}(x)) \]

        for each layer of the model separately.

        Args:
            x: model input to use in the gradient computations
            y: label tensor to compute gradients

        Returns:
            A dictionary containing the influence factors for each layer of the model,
            with the layer name as key.
        """
        if not self.is_fitted:
            raise ValueError(
                "Instance must be fitted before calling influence methods on it"
            )

        return self._solve_hvp_by_layer(
            self._loss_grad(x.to(self.model_device), y.to(self.model_device)),
            self.ekfac_representation,
            self.hessian_regularization,
        )

    def influences_from_factors_by_layer(
        self,
        z_test_factors: Dict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Computation of

        \[ \langle z_{\text{test_factors}},
            \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the case of up-weighting influence, resp.

        \[ \langle z_{\text{test_factors}},
            \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle \]

        for the perturbation type influence case for each layer of the model
        separately. The gradients are meant to be per sample of the batch $(x,
        y)$.

        Args:
            z_test_factors: pre-computed tensor, approximating
                $H^{-1}\nabla_{\theta} \ell(y_{\text{test}},
                f_{\theta}(x_{\text{test}}))$
            x: model input to use in the gradient computations
                $\nabla_{\theta}\ell(y, f_{\theta}(x))$,
                resp. $\nabla_{x}\nabla_{\theta}\ell(y, f_{\theta}(x))$
            y: label tensor to compute gradients
            mode: enum value of [InfluenceMode]
                [pydvl.influence.base_influence_function_model.InfluenceMode]

        Returns:
            A dictionary containing the influence of the data on the test data
            for each layer of the model, with the layer name as key.
        """
        if mode == InfluenceMode.Up:
            total_grad = self._loss_grad(
                x.to(self.model_device), y.to(self.model_device)
            )
            start_idx = 0
            influences = {}
            for layer_id, layer_z_test in z_test_factors.items():
                end_idx = start_idx + layer_z_test.shape[1]
                influences[layer_id] = (
                    layer_z_test.to(self.model_device)
                    @ total_grad[:, start_idx:end_idx].T
                )
                start_idx = end_idx
            return influences
        elif mode == InfluenceMode.Perturbation:
            total_mixed_grad = self._flat_loss_mixed_grad(
                x.to(self.model_device), y.to(self.model_device)
            )
            start_idx = 0
            influences = {}
            for layer_id, layer_z_test in z_test_factors.items():
                end_idx = start_idx + layer_z_test.shape[1]
                influences[layer_id] = torch.einsum(
                    "ia,j...a->ij...",
                    layer_z_test.to(self.model_device),
                    total_mixed_grad[:, start_idx:end_idx],
                )
                start_idx = end_idx
            return influences
        else:
            raise UnsupportedInfluenceModeException(mode)

    def _non_symmetric_values_by_layer(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        mode: InfluenceMode = InfluenceMode.Up,
    ) -> Dict[str, torch.Tensor]:
        """
        Similar to `_non_symmetric_values`, but computes the influence for each
        layer separately. Returns a dictionary containing the influence for each
        layer, with the layer name as key.
        """
        if mode == InfluenceMode.Up:
            if x_test.shape[0] <= x.shape[0]:
                fac = self.influence_factors_by_layer(x_test, y_test)
                values = self.influences_from_factors_by_layer(fac, x, y, mode=mode)
            else:
                fac = self.influence_factors_by_layer(x, y)
                values = self.influences_from_factors_by_layer(
                    fac, x_test, y_test, mode=mode
                )
        elif mode == InfluenceMode.Perturbation:
            fac = self.influence_factors_by_layer(x_test, y_test)
            values = self.influences_from_factors_by_layer(fac, x, y, mode=mode)
        else:
            raise UnsupportedInfluenceModeException(mode)
        return values

    def _symmetric_values_by_layer(
        self, x: torch.Tensor, y: torch.Tensor, mode: InfluenceMode
    ) -> Dict[str, torch.Tensor]:
        """
        Similar to `_symmetric_values`, but computes the influence for each layer
        separately. Returns a dictionary containing the influence for each layer,
        with the layer name as key.
        """
        grad = self._loss_grad(x, y)
        fac = self._solve_hvp_by_layer(
            grad, self.ekfac_representation, self.hessian_regularization
        )

        if mode == InfluenceMode.Up:
            values = {}
            start_idx = 0
            for layer_id, layer_fac in fac.items():
                end_idx = start_idx + layer_fac.shape[1]
                values[layer_id] = layer_fac @ grad[:, start_idx:end_idx].T
                start_idx = end_idx
        elif mode == InfluenceMode.Perturbation:
            values = self.influences_from_factors_by_layer(fac, x, y, mode=mode)
        else:
            raise UnsupportedInfluenceModeException(mode)
        return values

    def explore_hessian_regularization(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        regularization_values: List[float],
    ) -> Dict[float, Dict[str, torch.Tensor]]:
        """
        Efficiently computes the influence for input x and label y for each layer of the
        model, for different values of the hessian regularization parameter. This is done
        by computing the gradient of the loss function for the input x and label y only once
        and then solving the Hessian Vector Product for each regularization value. This is
        useful for finding the optimal regularization value and for exploring
        how robust the influence values are to changes in the regularization value.

        Args:
            x: model input to use in the gradient computations
            y: label tensor to compute gradients
            regularization_values: list of regularization values to use

        Returns:
            A dictionary containing with keys being the regularization values and values
            being dictionaries containing the influences for each layer of the model,
            with the layer name as key.
        """
        grad = self._loss_grad(x.to(self.model_device), y.to(self.model_device))
        influences_by_reg_value = {}
        for reg_value in regularization_values:
            reg_factors = self._solve_hvp_by_layer(
                grad, self.ekfac_representation, reg_value
            )
            values = {}
            start_idx = 0
            for layer_id, layer_fac in reg_factors.items():
                end_idx = start_idx + layer_fac.shape[1]
                values[layer_id] = layer_fac @ grad[:, start_idx:end_idx].T
                start_idx = end_idx
            influences_by_reg_value[reg_value] = values
        return influences_by_reg_value

    def to(self, device: torch.device):
        if self.is_fitted:
            self.ekfac_representation.to(device)
        return super().to(device)


class NystroemSketchInfluence(TorchInfluenceFunctionModel):
    r"""
    Given a model and training data, it uses a low-rank approximation of the Hessian
    (derived via random projection Nyström approximation) in combination with
    the [Sherman–Morrison–Woodbury
    formula](https://en.wikipedia.org/wiki/Woodbury_matrix_identity) to
    calculate the inverse of the Hessian Vector Product. More concrete, it
    computes a low-rank approximation

    \begin{align*}
        H_{\text{nys}} &= (H\Omega)(\Omega^TH\Omega)^{+}(H\Omega)^T \\\
                       &= U \Lambda U^T
    \end{align*}

    in factorized form and approximates the action of the inverse Hessian via

    \[ (H_{\text{nys}} + \lambda I)^{-1} = U(\Lambda+\lambda I)U^T +
        \frac{1}{\lambda}(I−UU^T). \]

    Args:
        model: A PyTorch model. The Hessian will be calculated with respect to
            this model's parameters.
        loss: A callable that takes the model's output and target as input and returns
              the scalar loss.
        hessian_regularization: Optional regularization parameter added
            to the Hessian-vector product for numerical stability.
        rank: rank of the low-rank approximation

    """

    low_rank_representation: LowRankProductRepresentation

    def __init__(
        self,
        model: torch.nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        hessian_regularization: float,
        rank: int,
    ):
        super().__init__(model, loss)
        self.hessian_regularization = hessian_regularization
        self.rank = rank

    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        regularized_eigenvalues = (
            self.low_rank_representation.eigen_vals + self.hessian_regularization
        )

        proj_rhs = self.low_rank_representation.projections.t() @ rhs.t()
        inverse_regularized_eigenvalues = 1.0 / regularized_eigenvalues
        result = self.low_rank_representation.projections @ (
            proj_rhs * inverse_regularized_eigenvalues.unsqueeze(-1)
        )

        if self.hessian_regularization > 0.0:
            result += (
                1.0
                / self.hessian_regularization
                * (rhs.t() - self.low_rank_representation.projections @ proj_rhs)
            )

        return result.t()

    @property
    def is_fitted(self):
        try:
            return self.low_rank_representation is not None
        except AttributeError:
            return False

    @log_duration(log_level=logging.INFO)
    def fit(self, data: DataLoader):
        self.low_rank_representation = model_hessian_nystroem_approximation(
            self.model, self.loss, data, self.rank
        )
        return self


class TorchOperatorGradientComposition(
    OperatorGradientComposition[
        torch.Tensor, TorchBatch, TorchOperator, TorchPerSampleGradientProvider
    ]
):
    """
    Representing a composable block that integrates an [TorchOperator]
    [pydvl.influence.torch.operator.base.TorchOperator] and
    a [TorchPerSampleGradientProvider]
    [pydvl.influence.torch.operator.gradient_provider.TorchPerSampleGradientProvider]

    This block is designed to be flexible, handling different computational modes via
    an abstract operator and gradient provider.
    """

    def __init__(self, op: TorchOperator, gp: TorchPerSampleGradientProvider):
        super().__init__(op, gp)

    def to(self, device: torch.device):
        self.gp = self.gp.to(device)
        self.op = self.op.to(device)
        return self


class TorchBlockMapper(
    BlockMapper[torch.Tensor, TorchBatch, TorchOperatorGradientComposition]
):
    """
    Class for mapping operations across multiple compositional blocks represented by
    instances of [TorchOperatorGradientComposition]
    [pydvl.influence.torch.influence_function_model.TorchOperatorGradientComposition].

    This class takes a dictionary of compositional blocks and applies their methods to
    batches or tensors, and aggregates the results.
    """

    def __init__(
        self, composable_block_dict: OrderedDict[str, TorchOperatorGradientComposition]
    ):
        super().__init__(composable_block_dict)

    def _split_to_blocks(
        self, z: torch.Tensor, dim: int = -1
    ) -> OrderedDict[str, torch.Tensor]:
        block_sizes = [bi.op.input_size for bi in self.composable_block_dict.values()]

        block_dict = OrderedDict(
            zip(
                list(self.composable_block_dict.keys()),
                torch.split(z, block_sizes, dim=dim),
            )
        )
        return block_dict

    def to(self, device: torch.device):
        self.composable_block_dict = OrderedDict(
            [(k, bi.to(device)) for k, bi in self.composable_block_dict.items()]
        )
        return self


class TorchComposableInfluence(
    ComposableInfluence[torch.Tensor, TorchBatch, DataLoader, TorchBlockMapper],
    ModelInfoMixin,
    ABC,
):
    def __init__(
        self,
        model: torch.nn.Module,
        block_structure: Union[
            BlockMode, OrderedDict[str, OrderedDict[str, torch.nn.Parameter]]
        ] = BlockMode.FULL,
        regularization: Optional[Union[float, Dict[str, Optional[float]]]] = None,
    ):
        if isinstance(block_structure, BlockMode):
            self.parameter_dict = ModelParameterDictBuilder(model).build(
                block_structure
            )
        else:
            self.parameter_dict = block_structure

        self._regularization_dict = self._build_regularization_dict(regularization)

        super().__init__(model)

    @property
    def block_names(self) -> List[str]:
        return list(self.parameter_dict.keys())

    @abstractmethod
    def with_regularization(
        self, regularization: Union[float, Dict[str, Optional[float]]]
    ) -> TorchComposableInfluence:
        pass

    def _build_regularization_dict(
        self, regularization: Optional[Union[float, Dict[str, Optional[float]]]]
    ) -> Dict[str, Optional[float]]:
        if regularization is None or isinstance(regularization, float):
            return {
                k: self._validate_regularization(k, regularization)
                for k in self.block_names
            }

        if set(regularization.keys()).issubset(set(self.block_names)):
            raise ValueError(
                f"The regularization must be a float or the keys of the regularization"
                f"dictionary must match a subset of"
                f"block names: \n {self.block_names}.\n Found not in block names: \n"
                f"{set(regularization.keys()).difference(set(self.block_names))}"
            )
        return {
            k: self._validate_regularization(k, regularization.get(k, None))
            for k in self.block_names
        }

    @staticmethod
    def _validate_regularization(
        block_name: str, value: Optional[float]
    ) -> Optional[float]:
        if isinstance(value, float) and value < 0.0:
            raise ValueError(
                f"The regularization for block '{block_name}' must be non-negative, "
                f"but found {value=}"
            )
        return value

    @abstractmethod
    def _create_block(
        self,
        block_params: Dict[str, torch.nn.Parameter],
        data: DataLoader,
        regularization: Optional[float],
    ) -> TorchOperatorGradientComposition:
        pass

    def _create_block_mapper(self, data: DataLoader) -> TorchBlockMapper:
        block_influence_dict = OrderedDict()
        for k, p in self.parameter_dict.items():
            reg = self._regularization_dict.get(k, None)
            reg = self._validate_regularization(k, reg)
            block_influence_dict[k] = self._create_block(p, data, reg).to(self.device)

        return TorchBlockMapper(block_influence_dict)

    @staticmethod
    def _create_batch(x: torch.Tensor, y: torch.Tensor) -> TorchBatch:
        return TorchBatch(x, y)

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        if hasattr(self, "block_mapper") and self.block_mapper is not None:
            self.block_mapper = self.block_mapper.to(device)
        return self


class InverseHarmonicMeanInfluence(TorchComposableInfluence):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: LossType,
        regularization: Union[float, Dict[str, Optional[float]]],
        block_structure: Union[
            BlockMode, OrderedDict[str, OrderedDict[str, torch.Tensor]]
        ] = BlockMode.FULL,
    ):
        super().__init__(model, block_structure, regularization=regularization)
        self.gradient_provider_factory = TorchPerSampleAutoGrad
        self.loss = loss

    @property
    def n_parameters(self):
        return super().n_parameters()

    @property
    def is_thread_safe(self) -> bool:
        return False

    @staticmethod
    def _validate_regularization(
        block_name: str, value: Optional[float]
    ) -> Optional[float]:
        if value is None or value <= 0.0:
            raise ValueError(
                f"The regularization for block '{block_name}' must be a positive float,"
                f"but found {value=}"
            )
        return value

    def _create_block(
        self,
        block_params: Dict[str, torch.nn.Parameter],
        data: DataLoader,
        regularization: Optional[float],
    ) -> TorchOperatorGradientComposition:
        assert regularization is not None
        op = InverseHarmonicMeanOperator(
            self.model,
            self.loss,
            data,
            regularization,
            self.gradient_provider_factory,
            restrict_to=block_params,
        )
        gp = self.gradient_provider_factory(
            self.model, self.loss, restrict_to=block_params
        )
        return TorchOperatorGradientComposition(op, gp)

    def with_regularization(
        self, regularization: Union[float, Dict[str, Optional[float]]]
    ) -> TorchComposableInfluence:
        self._regularization_dict = self._build_regularization_dict(regularization)
        for k, reg in self._regularization_dict.items():
            self.block_mapper.composable_block_dict[k].op.regularization = reg
        return self

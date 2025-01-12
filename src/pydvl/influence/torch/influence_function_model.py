"""
This module implements several implementations of [InfluenceFunctionModel]
[pydvl.influence.base_influence_function_model.InfluenceFunctionModel]
utilizing PyTorch.
"""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ...utils.progress import log_duration
from .. import InfluenceMode
from ..base_influence_function_model import (
    InfluenceFunctionModel,
    NotImplementedLayerRepresentationException,
)
from ..types import UnsupportedInfluenceModeException
from .base import (
    TorchComposableInfluence,
    TorchGradientProvider,
    TorchOperatorGradientComposition,
)
from .batch_operation import (
    BatchOperationType,
    GaussNewtonBatchOperation,
    HessianBatchOperation,
)
from .functional import (
    create_per_sample_gradient_function,
    create_per_sample_mixed_derivative_function,
    gauss_newton,
    hessian,
    operator_nystroem_approximation,
    operator_spectral_approximation,
)
from .operator import (
    CgOperator,
    DirectSolveOperator,
    GaussNewtonOperator,
    HessianOperator,
    InverseHarmonicMeanOperator,
    LissaOperator,
    LowRankOperator,
)
from .preconditioner import Preconditioner
from .util import (
    BlockMode,
    EkfacRepresentation,
    LossType,
    SecondOrderMode,
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
    "InverseHarmonicMeanInfluence",
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


class DirectInfluence(TorchComposableInfluence[DirectSolveOperator]):
    r"""
    Given a model and training data, it finds x such that \(Hx = b\),
    with \(H\) being the model hessian or Gauss-Newton matrix.


    Args:
        model: The model.
        loss: The loss function.
        regularization: The regularization parameter. In case a dictionary is provided,
            the keys must be a subset of the block identifiers.
        block_structure: The blocking structure, either a pre-defined enum or a
            custom block structure, see the information regarding
            [block-diagonal approximation][block-diagonal-approximation].
        second_order_mode: The second order mode, either `SecondOrderMode.HESSIAN` or
            `SecondOrderMode.GAUSS_NEWTON`.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: LossType,
        regularization: Optional[Union[float, Dict[str, Optional[float]]]] = None,
        block_structure: Union[BlockMode, OrderedDict[str, List[str]]] = BlockMode.FULL,
        second_order_mode: SecondOrderMode = SecondOrderMode.HESSIAN,
    ):
        super().__init__(
            model,
            block_structure=block_structure,
            regularization=regularization,
        )
        self.second_order_mode = second_order_mode
        self.loss = loss

    def with_regularization(
        self, regularization: Union[float, Dict[str, Optional[float]]]
    ) -> TorchComposableInfluence:
        """
        Update the regularization parameter.
        Args:
            regularization: Either a positive float or a dictionary with the
                block names as keys and the regularization values as values.

        Returns:
            The modified instance

        """
        self._regularization_dict = self._build_regularization_dict(regularization)
        for k, reg in self._regularization_dict.items():
            self.block_mapper.composable_block_dict[k].op.regularization = reg
        return self

    def _create_block(
        self,
        block_params: Dict[str, torch.nn.Parameter],
        data: DataLoader,
        regularization: Optional[float],
    ) -> TorchOperatorGradientComposition:
        gp = TorchGradientProvider(self.model, self.loss, restrict_to=block_params)

        if self.second_order_mode is SecondOrderMode.GAUSS_NEWTON:
            mat = gauss_newton(self.model, self.loss, data, restrict_to=block_params)
        else:
            mat = hessian(self.model, self.loss, data, restrict_to=block_params)

        op = DirectSolveOperator(
            mat, regularization=regularization, in_place_regularization=True
        )
        return TorchOperatorGradientComposition(op, gp)

    @property
    def is_thread_safe(self) -> bool:
        return False


class CgInfluence(TorchComposableInfluence[CgOperator]):
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
        regularization: Optional regularization parameter added
            to the Hessian-vector product for numerical stability.
        rtol: Maximum relative tolerance of result.
        atol: Absolute tolerance of result.
        maxiter: Maximum number of iterations. If None, defaults to 10*len(b).
        progress: If True, display progress bars for computing in the non-block mode
            (use_block_cg=False).
        preconditioner: Optional preconditioner to improve convergence of conjugate
            gradient method
        solve_simultaneously: If True, use a variant of conjugate gradient method to
            simultaneously solve for several right hand sides.
        warn_on_max_iteration: If True, logs a warning, if the desired tolerance is not
            achieved within `maxiter` iterations. If False, the log level for this
            information is `logging.DEBUG`
        block_structure: Union[BlockMode, OrderedDict[str, List[str]]] = BlockMode.FULL,
        second_order_mode: SecondOrderMode = SecondOrderMode.HESSIAN,

    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        regularization: Optional[Union[float, Dict[str, Optional[float]]]] = None,
        rtol: float = 1e-4,
        atol: float = 1e-6,
        maxiter: Optional[int] = None,
        progress: bool = False,
        precompute_grad: bool = False,
        preconditioner: Optional[Preconditioner] = None,
        solve_simultaneously: bool = False,
        warn_on_max_iteration: bool = True,
        block_structure: Union[BlockMode, OrderedDict[str, List[str]]] = BlockMode.FULL,
        second_order_mode: SecondOrderMode = SecondOrderMode.HESSIAN,
    ):
        super().__init__(model, block_structure, regularization)
        self.loss = loss
        self.warn_on_max_iteration = warn_on_max_iteration
        self.solve_simultaneously = solve_simultaneously
        self.preconditioner = preconditioner
        self.precompute_grad = precompute_grad
        self.progress = progress
        self.maxiter = maxiter
        self.atol = atol
        self.rtol = rtol
        self.second_order_mode = second_order_mode

    def with_regularization(
        self, regularization: Union[float, Dict[str, Optional[float]]]
    ) -> TorchComposableInfluence:
        """
        Update the regularization parameter.
        Args:
            regularization: Either a positive float or a dictionary with the
                block names as keys and the regularization values as values.

        Returns:
            The modified instance

        """
        self._regularization_dict = self._build_regularization_dict(regularization)
        for k, reg in self._regularization_dict.items():
            self.block_mapper.composable_block_dict[k].op.regularization = reg
        return self

    def _create_block(
        self,
        block_params: Dict[str, torch.nn.Parameter],
        data: DataLoader,
        regularization: Optional[float],
    ) -> TorchOperatorGradientComposition:
        op: Union[HessianOperator, GaussNewtonOperator]

        if self.second_order_mode is SecondOrderMode.GAUSS_NEWTON:
            op = GaussNewtonOperator(
                self.model, self.loss, data, restrict_to=block_params
            )
        else:
            op = HessianOperator(self.model, self.loss, data, restrict_to=block_params)

        preconditioner = None
        if self.preconditioner is not None:
            preconditioner = copy.copy(self.preconditioner).fit(op, regularization)

        cg_op = CgOperator(
            op,
            regularization,
            rtol=self.rtol,
            atol=self.atol,
            maxiter=self.maxiter,
            progress=self.progress,
            preconditioner=preconditioner,
            use_block_cg=self.solve_simultaneously,
            warn_on_max_iteration=self.warn_on_max_iteration,
        )
        gp = TorchGradientProvider(self.model, self.loss, restrict_to=block_params)
        return TorchOperatorGradientComposition(cg_op, gp)

    @property
    def is_thread_safe(self) -> bool:
        return False


class LissaInfluence(TorchComposableInfluence[LissaOperator[BatchOperationType]]):
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
        regularization: Optional regularization parameter added
            to the Hessian-vector product for numerical stability.
        maxiter: Maximum number of iterations.
        dampen: Dampening factor, defaults to 0 for no dampening.
        scale: Scaling factor, defaults to 10.
        rtol: tolerance to use for early stopping
        progress: If True, display progress bars.
        warn_on_max_iteration: If True, logs a warning, if the desired tolerance is not
            achieved within `maxiter` iterations. If False, the log level for this
            information is `logging.DEBUG`
        block_structure: The blocking structure, either a pre-defined enum or a
            custom block structure, see the information regarding
            [block-diagonal approximation][block-diagonal-approximation].
        second_order_mode: The second order mode, either `SecondOrderMode.HESSIAN` or
            `SecondOrderMode.GAUSS_NEWTON`.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        regularization: Optional[Union[float, Dict[str, Optional[float]]]] = None,
        maxiter: int = 1000,
        dampen: float = 0.0,
        scale: float = 10.0,
        rtol: float = 1e-4,
        progress: bool = False,
        warn_on_max_iteration: bool = True,
        block_structure: Union[BlockMode, OrderedDict[str, List[str]]] = BlockMode.FULL,
        second_order_mode: SecondOrderMode = SecondOrderMode.HESSIAN,
    ):
        super().__init__(model, block_structure, regularization)
        self.maxiter = maxiter
        self.progress = progress
        self.rtol = rtol
        self.scale = scale
        self.dampen = dampen
        self.loss = loss
        self.second_order_mode = second_order_mode
        self.warn_on_max_iteration = warn_on_max_iteration

    def with_regularization(
        self, regularization: Union[float, Dict[str, Optional[float]]]
    ) -> TorchComposableInfluence:
        """
        Update the regularization parameter.
        Args:
            regularization: Either a positive float or a dictionary with the
                block names as keys and the regularization values as values.

        Returns:
            The modified instance

        """
        self._regularization_dict = self._build_regularization_dict(regularization)
        for k, reg in self._regularization_dict.items():
            self.block_mapper.composable_block_dict[k].op.regularization = reg
        return self

    def _create_block(
        self,
        block_params: Dict[str, torch.nn.Parameter],
        data: DataLoader,
        regularization: Optional[float],
    ) -> TorchOperatorGradientComposition:
        gp = TorchGradientProvider(self.model, self.loss, restrict_to=block_params)
        batch_op: Union[GaussNewtonBatchOperation, HessianBatchOperation]
        if self.second_order_mode is SecondOrderMode.GAUSS_NEWTON:
            batch_op = GaussNewtonBatchOperation(
                self.model, self.loss, restrict_to=block_params
            )
        else:
            batch_op = HessianBatchOperation(
                self.model, self.loss, restrict_to=block_params
            )
        lissa_op = LissaOperator(
            batch_op,
            data,
            regularization,
            maxiter=self.maxiter,
            dampen=self.dampen,
            scale=self.scale,
            rtol=self.rtol,
            progress=self.progress,
            warn_on_max_iteration=self.warn_on_max_iteration,
        )
        return TorchOperatorGradientComposition(lissa_op, gp)

    @property
    def is_thread_safe(self) -> bool:
        return False


class ArnoldiInfluence(TorchComposableInfluence[LowRankOperator]):
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
        regularization: The regularization parameter. In case a dictionary is provided,
            the keys must be a subset of the block identifiers.
        rank: The number of eigenvalues and corresponding eigenvectors
            to compute. Represents the desired rank of the Hessian approximation.
        krylov_dimension: The number of Krylov vectors to use for the Lanczos method.
            Defaults to min(model's number of parameters,
            max(2 times rank + 1, 20)).
        tol: The stopping criteria for the Lanczos algorithm.
        max_iter: The maximum number of iterations for the Lanczos method.
        eigen_computation_on_gpu: If True, tries to execute the eigen pair approximation
            on the model's device
            via a cupy implementation. Ensure the model size or rank_estimate
            is appropriate for device memory.
            If False, the eigen pair approximation is executed on the CPU by the scipy
            wrapper to ARPACK.
        use_woodbury: If True, uses the [Sherman–Morrison–Woodbury
            formula](https://en.wikipedia.org/wiki/Woodbury_matrix_identity) for the
            computation of the inverse action, which is more precise but needs
            additional computation.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        regularization: Optional[Union[float, Dict[str, Optional[float]]]] = None,
        rank: int = 10,
        krylov_dimension: Optional[int] = None,
        tol: float = 1e-6,
        max_iter: Optional[int] = None,
        eigen_computation_on_gpu: bool = False,
        block_structure: Union[BlockMode, OrderedDict[str, List[str]]] = BlockMode.FULL,
        second_order_mode: SecondOrderMode = SecondOrderMode.HESSIAN,
        use_woodbury: bool = False,
    ):
        super().__init__(model, block_structure, regularization)
        self.use_woodbury = use_woodbury
        self.second_order_mode = second_order_mode
        self.loss = loss
        self.rank = rank
        self.tol = tol
        self.max_iter = max_iter
        self.krylov_dimension = krylov_dimension
        self.eigen_computation_on_gpu = eigen_computation_on_gpu

    def with_regularization(
        self, regularization: Union[float, Dict[str, Optional[float]]]
    ) -> TorchComposableInfluence:
        self._regularization_dict = self._build_regularization_dict(regularization)
        for k, reg in self._regularization_dict.items():
            self.block_mapper.composable_block_dict[k].op.regularization = reg
        return self

    def _create_block(
        self,
        block_params: Dict[str, torch.nn.Parameter],
        data: DataLoader,
        regularization: Optional[float],
    ) -> TorchOperatorGradientComposition:
        gp = TorchGradientProvider(self.model, self.loss, restrict_to=block_params)
        op: Union[HessianOperator, GaussNewtonOperator]
        if self.second_order_mode is SecondOrderMode.GAUSS_NEWTON:
            op = GaussNewtonOperator(
                self.model, self.loss, data, restrict_to=block_params
            )
        else:
            op = HessianOperator(self.model, self.loss, data, restrict_to=block_params)
        low_rank_representation = operator_spectral_approximation(
            op,
            self.rank,
            krylov_dimension=self.krylov_dimension,
            tol=self.tol,
            max_iter=self.max_iter,
            eigen_computation_on_gpu=self.eigen_computation_on_gpu,
        )
        low_rank_op = LowRankOperator(
            low_rank_representation, regularization, exact=self.use_woodbury
        )
        return TorchOperatorGradientComposition(low_rank_op, gp)

    @property
    def is_thread_safe(self) -> bool:
        return False


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


class NystroemSketchInfluence(TorchComposableInfluence[LowRankOperator]):
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
        regularization: Optional regularization parameter added
            to the Hessian-vector product for numerical stability.
        rank: rank of the low-rank approximation

    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        regularization: Union[float, Dict[str, float]],
        rank: int,
        block_structure: Union[BlockMode, OrderedDict[str, List[str]]] = BlockMode.FULL,
        second_order_mode: SecondOrderMode = SecondOrderMode.HESSIAN,
    ):
        super().__init__(
            model,
            block_structure,
            regularization=cast(
                Union[float, Dict[str, Optional[float]]], regularization
            ),
        )
        self.second_order_mode = second_order_mode
        self.rank = rank
        self.loss = loss

    def with_regularization(
        self, regularization: Union[float, Dict[str, Optional[float]]]
    ) -> TorchComposableInfluence:
        self._regularization_dict = self._build_regularization_dict(regularization)
        for k, reg in self._regularization_dict.items():
            self.block_mapper.composable_block_dict[k].op.regularization = reg
        return self

    def _create_block(
        self,
        block_params: Dict[str, torch.nn.Parameter],
        data: DataLoader,
        regularization: Optional[float],
    ) -> TorchOperatorGradientComposition:
        assert regularization is not None
        regularization = cast(float, regularization)

        op: Union[HessianOperator, GaussNewtonOperator]

        if self.second_order_mode is SecondOrderMode.HESSIAN:
            op = HessianOperator(self.model, self.loss, data, restrict_to=block_params)
        elif self.second_order_mode is SecondOrderMode.GAUSS_NEWTON:
            op = GaussNewtonOperator(
                self.model, self.loss, data, restrict_to=block_params
            )
        else:
            raise ValueError(f"Unsupported second order mode: {self.second_order_mode}")

        low_rank_repr = operator_nystroem_approximation(op, self.rank)
        low_rank_op = LowRankOperator(low_rank_repr, regularization)
        gp = TorchGradientProvider(self.model, self.loss, restrict_to=block_params)
        return TorchOperatorGradientComposition(low_rank_op, gp)

    @property
    def is_thread_safe(self) -> bool:
        return False


class InverseHarmonicMeanInfluence(
    TorchComposableInfluence[InverseHarmonicMeanOperator]
):
    r"""
    This implementation replaces the inverse Hessian matrix in the influence computation
    with an approximation of the inverse Gauss-Newton vector product.

    Viewing the damped Gauss-newton matrix

    \begin{align*}
        G_{\lambda}(\theta) &=
        \frac{1}{N}\sum_{i}^N\nabla_{\theta}\ell (x_i,y_i; \theta)
            \nabla_{\theta}\ell (x_i, y_i; \theta)^t + \lambda \operatorname{I}, \\\
        \ell(x,y; \theta) &= \text{loss}(\text{model}(x; \theta), y)
    \end{align*}

    as an arithmetic mean of the rank-$1$ updates, this implementation replaces it with
    the harmonic mean of the rank-$1$ updates, i.e.

    $$ \tilde{G}_{\lambda}(\theta) =
        \left(N \cdot \sum_{i=1}^N  \left( \nabla_{\theta}\ell (x_i,y_i; \theta)
            \nabla_{\theta}\ell (x_i,y_i; \theta)^t +
            \lambda \operatorname{I}\right)^{-1}
            \right)^{-1}$$

    and uses the matrix

    $$ \tilde{G}_{\lambda}^{-1}(\theta)$$

    instead of the inverse Hessian.

    In other words, it switches the order of summation and inversion, which resolves
    to the `inverse harmonic mean` of the rank-$1$ updates. The results are averaged
    over the batches provided by the data loader.

    The inverses of the rank-$1$ updates are not calculated explicitly,
    but instead a vectorized version of the
    [Sherman–Morrison formula](
    https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula)
    is applied.

    For more information,
    see [Inverse Harmonic Mean][inverse-harmonic-mean].

    Args:
        model: The model.
        loss: The loss function.
        regularization: The regularization parameter. In case a dictionary is provided,
            the keys must match the blocking structure and the specification must be
            complete, so every block needs a positive regularization value, which
            differs from the description in
            [block-diagonal approximation][block-diagonal-approximation].
        block_structure: The blocking structure, either a pre-defined enum or a
            custom block structure, see the information regarding
            [block-diagonal approximation][block-diagonal-approximation].
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: LossType,
        regularization: Union[float, Dict[str, float]],
        block_structure: Union[BlockMode, OrderedDict[str, List[str]]] = BlockMode.FULL,
    ):
        super().__init__(
            model,
            block_structure,
            regularization=cast(
                Union[float, Dict[str, Optional[float]]], regularization
            ),
        )
        self.loss = loss

    @property
    def n_parameters(self):
        return sum(block.op.input_size for _, block in self.block_mapper.items())

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
            restrict_to=block_params,
        )
        gp = TorchGradientProvider(self.model, self.loss, restrict_to=block_params)
        return TorchOperatorGradientComposition(op, gp)

    def with_regularization(
        self, regularization: Union[float, Dict[str, Optional[float]]]
    ) -> TorchComposableInfluence:
        """
        Update the regularization parameter.
        Args:
            regularization: Either a positive float or a dictionary with the
                block names as keys and the regularization values as values.

        Returns:
            The modified instance

        """
        self._regularization_dict = self._build_regularization_dict(regularization)
        for k, reg in self._regularization_dict.items():
            self.block_mapper.composable_block_dict[k].op.regularization = reg
        return self

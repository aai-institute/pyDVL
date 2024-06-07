import logging
from typing import Callable, Dict, Generic, Optional, Tuple

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import TensorDictOperator, TensorOperator, TorchBatch
from .batch_operation import (
    BatchOperationType,
    ChunkAveraging,
    GaussNewtonBatchOperation,
    HessianBatchOperation,
    InverseHarmonicMeanBatchOperation,
    PointAveraging,
    TensorAveragingType,
)
from .functional import LowRankProductRepresentation

logger = logging.getLogger(__name__)


class _AveragingBatchOperator(
    TensorDictOperator, Generic[BatchOperationType, TensorAveragingType]
):
    """
    Class for aggregating batch operations over a dataset using a provided data loader
    and aggregator.

    This class facilitates the application of a batch operation across multiple batches
    of data, aggregating the results using a specified sequence aggregator.

    Attributes:
        batch_operation: The batch operation to apply.
        dataloader: The data loader providing batches of data.
        averaging: The sequence aggregator to aggregate the results of the batch
            operations.
    """

    def __init__(
        self,
        batch_operation: BatchOperationType,
        dataloader: DataLoader,
        averager: TensorAveragingType,
    ):
        self.batch_operation = batch_operation
        self.dataloader = dataloader
        self.averaging = averager

    @property
    def input_dict_structure(self) -> Dict[str, Tuple[int, ...]]:
        return self.batch_operation.input_dict_structure

    def _apply_to_dict(self, mat: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        tensor_dicts = (
            self.batch_operation.apply_to_dict(TorchBatch(x, y), mat)
            for x, y in self.dataloader
        )
        dict_averaging = self.averaging.as_dict_averaging()
        result: Dict[str, torch.Tensor] = dict_averaging(tensor_dicts)
        return result

    @property
    def device(self):
        return self.batch_operation.device

    @property
    def dtype(self):
        return self.batch_operation.dtype

    def to(self, device: torch.device):
        self.batch_operation = self.batch_operation.to(device)
        return self

    @property
    def input_size(self):
        return self.batch_operation.input_size

    def _apply_to_mat(self, mat: torch.Tensor) -> torch.Tensor:
        return self._apply_to_vec(mat)

    def _apply_to_vec(self, vec: torch.Tensor) -> torch.Tensor:
        tensors = (
            self.batch_operation.apply(
                TorchBatch(x.to(self.device), y.to(self.device)), vec.to(self.device)
            )
            for x, y in self.dataloader
        )

        return self.averaging(tensors)


class GaussNewtonOperator(
    _AveragingBatchOperator[GaussNewtonBatchOperation, PointAveraging]
):
    r"""
    Given a model and loss function computes the Gauss-Newton vector or matrix product
    with respect to the model parameters on a batch, i.e.

    \begin{align*}
        G(\text{model}, \text{loss}, b, \theta) &\cdot v, \\\
        G(\text{model}, \text{loss}, b, \theta) &=
        \frac{1}{|b|}\sum_{(x, y) \in b}\nabla_{\theta}\ell (x,y; \theta)
            \nabla_{\theta}\ell (x,y; \theta)^t, \\\
        \ell(x,y; \theta) &= \text{loss}(\text{model}(x; \theta), y)
    \end{align*}

    where model is a [torch.nn.Module][torch.nn.Module] and $v$ is a vector or matrix,
    and average the results over the batches provided by the data loader.

    Args:
        model: The model.
        loss: The loss function.
        dataloader: The data loader providing batches of data.
        restrict_to: The parameters to restrict the differentiation to,
            i.e. the corresponding sub-matrix of the Jacobian. If None, the full
            Jacobian is used. Make sure the input matches the corrct dimension, i.e. the
            last dimension must be equal to the property `input_size`.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        dataloader: DataLoader,
        restrict_to: Optional[Dict[str, nn.Parameter]] = None,
    ):
        batch_op = GaussNewtonBatchOperation(
            model,
            loss,
            restrict_to=restrict_to,
        )
        averaging = PointAveraging()
        super().__init__(batch_op, dataloader, averaging)


class HessianOperator(_AveragingBatchOperator[HessianBatchOperation, ChunkAveraging]):
    r"""
    Given a model and loss function computes the Hessian vector or matrix product
    with respect to the model parameters for a given batch, i.e.

    \begin{align*}
        &\nabla^2_{\theta} L(b;\theta) \cdot v \\\
        &L(b;\theta) = \left( \frac{1}{|b|} \sum_{(x,y) \in b}
        \text{loss}(\text{model}(x; \theta), y)\right),
    \end{align*}

    where model is a [torch.nn.Module][torch.nn.Module] and $v$ is a vector or matrix,
    and average the results over the batches provided by the data loader.

    Args:
        model: The model.
        loss: The loss function.
        dataloader: The data loader providing batches of data.
        restrict_to: The parameters to restrict the second order differentiation to,
            i.e. the corresponding sub-matrix of the Hessian. If None, the full Hessian
            is used. Make sure the input matches the corrct dimension, i.e. the
            last dimension must be equal to the property `input_size`.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        dataloader: DataLoader,
        restrict_to: Optional[Dict[str, nn.Parameter]] = None,
    ):
        batch_op = HessianBatchOperation(model, loss, restrict_to=restrict_to)
        averaging = ChunkAveraging()
        super().__init__(batch_op, dataloader, averaging)


class InverseHarmonicMeanOperator(
    _AveragingBatchOperator[InverseHarmonicMeanBatchOperation, PointAveraging]
):
    r"""
    Given a model and loss function computes an approximation of the inverse
    Gauss-Newton vector or matrix product per batch and averages the results.

    Viewing the damped Gauss-newton matrix

    \begin{align*}
        G_{\lambda}(\text{model}, \text{loss}, b, \theta) &=
        \frac{1}{|b|}\sum_{(x, y) \in b}\nabla_{\theta}\ell (x,y; \theta)
            \nabla_{\theta}\ell (x,y; \theta)^t + \lambda \operatorname{I}, \\\
        \ell(x,y; \theta) &= \text{loss}(\text{model}(x; \theta), y)
    \end{align*}

    as an arithmetic mean of the rank-$1$ updates, this operator replaces it with
    the harmonic mean of the rank-$1$ updates, i.e.

    $$ \tilde{G}_{\lambda}(\text{model}, \text{loss}, b, \theta) =
        \left(n \sum_{(x, y) \in b}  \left( \nabla_{\theta}\ell (x,y; \theta)
            \nabla_{\theta}\ell (x,y; \theta)^t + \lambda \operatorname{I}\right)^{-1}
            \right)^{-1}$$

    and computes

    $$ \tilde{G}_{\lambda}^{-1}(\text{model}, \text{loss}, b, \theta)
    \cdot v.$$

    for any given batch $b$,
    where model is a [torch.nn.Module][torch.nn.Module] and $v$ is a vector or matrix.

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
        dataloader: The data loader providing batches of data.
        restrict_to: The parameters to restrict the differentiation to,
            i.e. the corresponding sub-matrix of the Jacobian. If None, the full
            Jacobian is used. Make sure the input matches the corrct dimension, i.e. the
            last dimension must be equal to the property `input_size`.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        dataloader: DataLoader,
        regularization: float,
        restrict_to: Optional[Dict[str, nn.Parameter]] = None,
    ):
        if regularization <= 0:
            raise ValueError("regularization must be positive")

        self._regularization = regularization

        batch_op = InverseHarmonicMeanBatchOperation(
            model,
            loss,
            regularization,
            restrict_to=restrict_to,
        )
        averaging = PointAveraging()
        super().__init__(batch_op, dataloader, averaging)

    @property
    def regularization(self):
        return self._regularization

    @regularization.setter
    def regularization(self, value: float):
        if value <= 0:
            raise ValueError("regularization must be positive")
        self._regularization = value
        self.batch_operation.regularization = value


class DirectSolveOperator(TensorOperator):
    r"""
    Given a matrix $A$ and an optional regularization parameter $\lambda$,
    computes the solution of the system $(A+\lambda I)x = b$, where $b$ is a
    vector or a matrix. Internally, it uses the routine
    [torch.linalg.solve][torch.linalg.solve].

    Args:
        matrix: the system matrix
        regularization: the regularization parameter
        in_place_regularization: If True, the input matrix is modified in-place, by
            adding the regularization value to the diagonal.

    """

    def __init__(
        self,
        matrix: torch.Tensor,
        regularization: Optional[float] = None,
        in_place_regularization: bool = False,
    ):
        if regularization is None:
            self.matrix = matrix
        else:
            self.matrix = self._update_diagonal(
                matrix if in_place_regularization else matrix.clone(), regularization
            )
        self._regularization = regularization

    @staticmethod
    def _update_diagonal(matrix: torch.Tensor, value: float) -> torch.Tensor:
        diag_indices = torch.arange(matrix.shape[-1], device=matrix.device)
        matrix[diag_indices, diag_indices] += value
        return matrix

    @property
    def regularization(self):
        return self._regularization

    @regularization.setter
    def regularization(self, value: float):
        if value <= 0:
            raise ValueError("regularization must be positive")
        old_value = self._regularization
        if old_value is None:
            self.matrix = self._update_diagonal(self.matrix, value)
        else:
            self.matrix = self._update_diagonal(self.matrix, value - old_value)

        self._regularization = value

    @property
    def device(self):
        return self.matrix.device

    @property
    def dtype(self):
        return self.matrix.dtype

    def to(self, device: torch.device):
        self.matrix = self.matrix.to(device)
        return self

    def _apply_to_vec(self, vec: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(self.matrix, vec.t()).t()

    def _apply_to_mat(self, mat: torch.Tensor) -> torch.Tensor:
        return self._apply_to_vec(mat)

    @property
    def input_size(self) -> int:
        result: int = self.matrix.shape[-1]
        return result


class LissaOperator(TensorOperator, Generic[BatchOperationType]):
    r"""
    Uses LISSA, Linear time Stochastic Second-Order Algorithm, to iteratively
    approximate the solution of the system \((A + \lambda I)x = b\).
    This is done with the update

    \[(A + \lambda I)^{-1}_{j+1} b = b + (I - d) \ (A + \lambda I) -
        \frac{(A + \lambda I)^{-1}_j b}{s},\]

    where \(I\) is the identity matrix, \(d\) is a dampening term and \(s\) a scaling
    factor that are applied to help convergence. For details,
    see [Linear time Stochastic Second-Order Approximation (LiSSA)]
    [linear-time-stochastic-second-order-approximation-lissa]

    Args:
        batch_operation: The `BatchOperation` representing the action of A on a batch
            of the data loader.
        data: a pytorch dataloader
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
    """

    def __init__(
        self,
        batch_operation: BatchOperationType,
        data: DataLoader,
        regularization: Optional[float] = None,
        maxiter: int = 1000,
        dampen: float = 0.0,
        scale: float = 10.0,
        rtol: float = 1e-4,
        progress: bool = False,
        warn_on_max_iteration: bool = True,
    ):

        if regularization is not None and regularization < 0:
            raise ValueError("regularization must be non-negative")

        self.data = data
        self.warn_on_max_iteration = warn_on_max_iteration
        self.progress = progress
        self.rtol = rtol
        self.scale = scale
        self.dampen = dampen
        self.maxiter = maxiter
        self.batch_operation = batch_operation
        self._regularization = regularization

    @property
    def regularization(self):
        return self._regularization

    @regularization.setter
    def regularization(self, value: float):
        if value < 0:
            raise ValueError("regularization must be non-negative")
        self._regularization = value

    @property
    def device(self):
        return self.batch_operation.device

    @property
    def dtype(self):
        return self.batch_operation.dtype

    def to(self, device: torch.device):
        self.batch_operation = self.batch_operation.to(device)
        return self

    def _reg_apply(self, batch: TorchBatch, h: torch.Tensor):
        result = self.batch_operation.apply(batch, h)
        if self.regularization is not None:
            result += self.regularization * h
        return result

    def _lissa_step(self, h: torch.Tensor, rhs: torch.Tensor, batch: TorchBatch):
        result = rhs + (1 - self.dampen) * h - self._reg_apply(batch, h) / self.scale
        if result.requires_grad:
            result = result.detach()
        return result

    def _apply_to_vec(self, vec: torch.Tensor) -> torch.Tensor:
        h_estimate = torch.clone(vec)
        shuffled_training_data = DataLoader(
            self.data.dataset,
            self.data.batch_size,
            shuffle=True,
        )
        is_converged = False
        for k in tqdm(
            range(self.maxiter), disable=not self.progress, desc="Lissa iteration"
        ):
            x, y = next(iter(shuffled_training_data))

            residual = self._lissa_step(h_estimate, vec, TorchBatch(x, y)) - h_estimate
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
                is_converged = True
                break

        if not is_converged:
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

    def _apply_to_mat(self, mat: torch.Tensor) -> torch.Tensor:
        return self._apply_to_vec(mat)

    @property
    def input_size(self) -> int:
        return self.batch_operation.input_size


class LowRankOperator(TensorOperator):
    r"""
    Given a low rank representation of a matrix

    $$ A = V D V^T$$

    with a diagonal matrix $D$ and an optional regularization parameter $\lambda$,
    computes

    $$ (V D V^T+\lambda I)^{-1}b$$.

    Depending on the value of the `exact` flag, the inverse action is computed exactly
    using the [Sherman–Morrison–Woodbury formula]
    (https://en.wikipedia.org/wiki/Woodbury_matrix_identity). If `exact` is set to
    `False`, the inverse action is approximated by

    $$ V^T(D+\lambda I)^{-1}Vb$$

    Args:
    """

    def __init__(
        self,
        low_rank_representation: LowRankProductRepresentation,
        regularization: float,
        exact: bool = True,
    ):

        if exact and (regularization is None or regularization <= 0):
            raise ValueError("regularization must be positive when exact=True")
        elif regularization is not None and regularization < 0:
            raise ValueError("regularization must be non-negative")

        self._regularization = regularization
        self._exact = exact
        self._low_rank_representation = low_rank_representation

    @property
    def exact(self):
        return self._exact

    @property
    def regularization(self):
        return self._regularization

    @regularization.setter
    def regularization(self, value: float):
        if value < 0:
            raise ValueError("regularization must be non-negative")
        self._regularization = value

    @property
    def device(self):
        return self._low_rank_representation.device

    @property
    def dtype(self):
        return self._low_rank_representation.dtype

    def to(self, device: torch.device):
        self._low_rank_representation = self._low_rank_representation.to(device)
        return self

    def _apply_to_vec(self, vec: torch.Tensor) -> torch.Tensor:

        if vec.ndim == 1:
            return self._apply_to_mat(vec.unsqueeze(0)).squeeze()

        return self._apply_to_mat(vec)

    def _apply_to_mat(self, mat: torch.Tensor) -> torch.Tensor:

        regularized_eigen_vals = self._low_rank_representation.eigen_vals.clone()

        if self.regularization is not None:
            regularized_eigen_vals += self.regularization

        proj_rhs = self._low_rank_representation.projections.t() @ mat.t()
        inverse_regularized_eigenvalues = 1.0 / regularized_eigen_vals
        result = self._low_rank_representation.projections @ (
            proj_rhs * inverse_regularized_eigenvalues.unsqueeze(-1)
        )

        if self._exact:
            result += (
                1.0
                / self.regularization
                * (mat.t() - self._low_rank_representation.projections @ proj_rhs)
            )

        return result.t()

    @property
    def input_size(self) -> int:
        result: int = self._low_rank_representation.projections.shape[0]
        return result

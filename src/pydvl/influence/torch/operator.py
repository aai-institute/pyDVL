from typing import Callable, Dict, Generator, Generic, Optional, Type, Union

import torch
from torch import nn as nn
from torch.utils.data import DataLoader

from ..array import LazyChunkSequence, SequenceAggregator
from .base import (
    GradientProviderFactoryType,
    TorchAutoGrad,
    TorchBatch,
    TorchGradientProvider,
    TorchOperator,
)
from .batch_operation import (
    BatchOperationType,
    GaussNewtonBatchOperation,
    HessianBatchOperation,
    InverseHarmonicMeanBatchOperation,
)
from .util import TorchChunkAverageAggregator, TorchPointAverageAggregator


class AggregateBatchOperator(TorchOperator, Generic[BatchOperationType]):
    """
    Class for aggregating batch operations over a dataset using a provided data loader
    and aggregator.

    This class facilitates the application of a batch operation across multiple batches
    of data, aggregating the results using a specified sequence aggregator.

    Attributes:
        batch_operation: The batch operation to apply.
        dataloader: The data loader providing batches of data.
        aggregator: The sequence aggregator to aggregate the results of the batch
            operations.
    """

    def __init__(
        self,
        batch_operation: BatchOperationType,
        dataloader: DataLoader,
        aggregator: SequenceAggregator[torch.Tensor],
    ):
        self.batch_operation = batch_operation
        self.dataloader = dataloader
        self.aggregator = aggregator

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

    def _apply_to_vec(self, vec: torch.Tensor) -> torch.Tensor:
        return self._apply(vec, self.batch_operation.apply_to_vec)

    def apply_to_mat(self, mat: torch.Tensor) -> torch.Tensor:
        """
        Applies the operator to a matrix.
        Args:
            mat: A matrix to apply the operator to. The last dimension is
                assumed to be consistent to the operation, i.e. it must equal
                to the property `input_size`.

        Returns:
            A matrix of shape $(N, \text{input_size})$, given the shape of mat is
                $(N, \text{input_size})$

        """
        return self._apply(mat, self.batch_operation.apply_to_mat)

    def _apply(
        self,
        z: torch.Tensor,
        batch_ops: Callable[[TorchBatch, torch.Tensor], torch.Tensor],
    ):
        def tensor_gen_factory() -> Generator[torch.Tensor, None, None]:
            return (
                batch_ops(
                    TorchBatch(x.to(self.device), y.to(self.device)), z.to(self.device)
                )
                for x, y in self.dataloader
            )

        lazy_tensor_sequence = LazyChunkSequence(
            tensor_gen_factory, len_generator=len(self.dataloader)
        )
        return self.aggregator(lazy_tensor_sequence)


class GaussNewtonOperator(AggregateBatchOperator[GaussNewtonBatchOperation]):
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
        gradient_provider_factory: Union[
            GradientProviderFactoryType,
            Type[TorchGradientProvider],
        ] = TorchAutoGrad,
        restrict_to: Optional[Dict[str, nn.Parameter]] = None,
    ):
        batch_op = GaussNewtonBatchOperation(
            model,
            loss,
            gradient_provider_factory=gradient_provider_factory,
            restrict_to=restrict_to,
        )
        aggregator = TorchPointAverageAggregator()
        super().__init__(batch_op, dataloader, aggregator)


class HessianOperator(AggregateBatchOperator[HessianBatchOperation]):
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
        reverse_only: If True only the reverse mode is used in the autograd computation.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        dataloader: DataLoader,
        restrict_to: Optional[Dict[str, nn.Parameter]] = None,
        reverse_only: bool = True,
    ):
        batch_op = HessianBatchOperation(
            model, loss, restrict_to=restrict_to, reverse_only=reverse_only
        )
        aggregator = TorchChunkAverageAggregator()
        super().__init__(batch_op, dataloader, aggregator)


class InverseHarmonicMeanOperator(
    AggregateBatchOperator[InverseHarmonicMeanBatchOperation]
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
        gradient_provider_factory: Union[
            GradientProviderFactoryType,
            Type[TorchGradientProvider],
        ] = TorchAutoGrad,
        restrict_to: Optional[Dict[str, nn.Parameter]] = None,
    ):
        if regularization <= 0:
            raise ValueError("regularization must be positive")

        self._regularization = regularization

        batch_op = InverseHarmonicMeanBatchOperation(
            model,
            loss,
            regularization,
            gradient_provider_factory=gradient_provider_factory,
            restrict_to=restrict_to,
        )
        aggregator = TorchPointAverageAggregator(weighted=False)
        super().__init__(batch_op, dataloader, aggregator)

    @property
    def regularization(self):
        return self._regularization

    @regularization.setter
    def regularization(self, value: float):
        if value <= 0:
            raise ValueError("regularization must be positive")
        self._regularization = value
        self.batch_operation.regularization = value

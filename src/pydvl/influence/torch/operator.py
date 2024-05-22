from typing import Callable, Generator, Union, Type, Optional, Dict

import torch
from torch import nn as nn
from torch.utils.data import DataLoader

from ..array import SequenceAggregator, LazyChunkSequence
from .base import TorchOperator, TorchBatch, \
    GradientProviderFactoryType, TorchPerSampleGradientProvider, TorchPerSampleAutoGrad
from .batch_operation import BatchOperation, \
    GaussNewtonBatchOperation, HessianBatchOperation, InverseHarmonicMeanBatchOperation
from .util import TorchPointAverageAggregator, \
    TorchChunkAverageAggregator


class AggregateBatchOperator(TorchOperator):
    def __init__(
        self,
        batch_operation: BatchOperation,
        dataloader: DataLoader,
        aggregator: SequenceAggregator[torch.Tensor],
    ):
        self.batch_operation = batch_operation
        self.dataloader = dataloader
        self.aggregator = aggregator
        super().__init__(self.batch_operation.regularization)

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
    def regularization(self):
        return self._regularization

    @regularization.setter
    def regularization(self, value: float):
        self._regularization = value
        self.batch_operation.regularization = value

    @property
    def input_size(self):
        return self.batch_operation.n_parameters

    def _apply_to_vec(self, vec: torch.Tensor) -> torch.Tensor:
        return self._apply(vec, self.batch_operation.apply_to_vec)

    def apply_to_mat(self, mat: torch.Tensor) -> torch.Tensor:
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


class GaussNewtonOperator(AggregateBatchOperator):
    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        dataloader: DataLoader,
        gradient_provider_factory: Union[
            GradientProviderFactoryType,
            Type[TorchPerSampleGradientProvider],
        ] = TorchPerSampleAutoGrad,
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


class HessianOperator(AggregateBatchOperator):
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


class InverseHarmonicMeanOperator(AggregateBatchOperator):
    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        dataloader: DataLoader,
        regularization: float,
        gradient_provider_factory: Union[
            GradientProviderFactoryType,
            Type[TorchPerSampleGradientProvider],
        ] = TorchPerSampleAutoGrad,
        restrict_to: Optional[Dict[str, nn.Parameter]] = None,
    ):
        batch_op = InverseHarmonicMeanBatchOperation(
            model,
            loss,
            regularization,
            gradient_provider_factory=gradient_provider_factory,
            restrict_to=restrict_to,
        )
        aggregator = TorchPointAverageAggregator(weighted=False)
        super().__init__(batch_op, dataloader, aggregator)

from typing import Callable, Dict, Optional, Type, Union

import torch
from torch import nn as nn
from torch.utils.data import DataLoader

from ..util import TorchPointAverageAggregator
from .base import AggregateBatchOperator, TorchOperator
from .batch_operation import InverseHarmonicMeanBatchOperation
from .gradient_provider import (
    GradientProviderFactoryType,
    TorchPerSampleAutoGrad,
    TorchPerSampleGradientProvider,
)

__all__ = ["InverseHarmonicMeanOperator"]


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

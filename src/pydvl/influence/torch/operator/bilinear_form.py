from typing import TYPE_CHECKING, Optional, cast

import torch

from ...types import BilinearForm
from ..util import TorchBatch
from .gradient_provider import TorchPerSampleGradientProvider

if TYPE_CHECKING:
    from .base import TorchOperator


class OperatorBilinearForm(
    BilinearForm[torch.Tensor, TorchBatch, TorchPerSampleGradientProvider]
):
    def __init__(
        self,
        operator: "TorchOperator",
    ):
        self.operator = operator

    def inner_product(
        self, left: torch.Tensor, right: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if right is None:
            right = left
        if left.shape[0] <= right.shape[0]:
            return self._inner_product(left, right)
        return self._inner_product(right, left).T

    def _inner_product(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_result = self.operator.apply_to_mat(left)

        if left_result.ndim == right.ndim and left.shape[-1] == right.shape[-1]:
            return left_result @ right.T

        return torch.einsum("ia,j...a->ij...", left_result, right)

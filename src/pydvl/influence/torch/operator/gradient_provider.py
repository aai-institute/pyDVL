from abc import ABC, abstractmethod
from typing import Dict, Callable, Optional

import torch
from torch.func import functional_call

from ..functional import (
    create_per_sample_gradient_function,
    create_per_sample_mixed_derivative_function,
    create_matrix_jacobian_product_function,
)

from ..util import (
    flatten_dimensions,
    LossType,
    TorchBatch,
    ModelParameterDictBuilder,
    BlockMode,
)

from ...types import PerSampleGradientProvider


class TorchPerSampleGradientProvider(
    PerSampleGradientProvider[TorchBatch, torch.Tensor], ABC
):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: LossType,
        restrict_to: Optional[Dict[str, torch.nn.Parameter]],
    ):
        self.loss = loss
        self.model = model

        if restrict_to is None:
            restrict_to = ModelParameterDictBuilder(model).build(BlockMode.FULL)

        self.params_to_restrict_to = restrict_to

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        self.params_to_restrict_to = {
            k: p.detach()
            for k, p in self.model.named_parameters()
            if k in self.params_to_restrict_to
        }
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @abstractmethod
    def _per_sample_gradient_dict(self, batch: TorchBatch) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _per_sample_mixed_gradient_dict(
        self, batch: TorchBatch
    ) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _matrix_jacobian_product(
        self,
        batch: TorchBatch,
        g: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def _detach_dict(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: g.detach() if g.requires_grad else g for k, g in tensor_dict.items()}

    def per_sample_gradient_dict(self, batch: TorchBatch) -> Dict[str, torch.Tensor]:
        gradient_dict = self._per_sample_gradient_dict(batch.to(self.device))
        return self._detach_dict(gradient_dict)

    def per_sample_mixed_gradient_dict(
        self, batch: TorchBatch
    ) -> Dict[str, torch.Tensor]:
        gradient_dict = self._per_sample_mixed_gradient_dict(batch.to(self.device))
        return self._detach_dict(gradient_dict)

    def matrix_jacobian_product(
        self,
        batch: TorchBatch,
        g: torch.Tensor,
    ) -> torch.Tensor:
        result = self._matrix_jacobian_product(batch.to(self.device), g.to(self.device))
        if result.requires_grad:
            result = result.detach()
        return result

    def per_sample_flat_gradient(self, batch: TorchBatch) -> torch.Tensor:
        return flatten_dimensions(
            self.per_sample_gradient_dict(batch).values(), shape=(batch.x.shape[0], -1)
        )

    def per_sample_flat_mixed_gradient(self, batch: TorchBatch) -> torch.Tensor:
        shape = (*batch.x.shape, -1)
        return flatten_dimensions(
            self.per_sample_mixed_gradient_dict(batch).values(), shape=shape
        )


class TorchPerSampleAutoGrad(TorchPerSampleGradientProvider):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: LossType,
        restrict_to: Optional[Dict[str, torch.nn.Parameter]] = None,
    ):
        super().__init__(model, loss, restrict_to)
        self._per_sample_gradient_function = create_per_sample_gradient_function(
            model, loss
        )
        self._per_sample_mixed_gradient_func = (
            create_per_sample_mixed_derivative_function(model, loss)
        )

    def _compute_loss(
        self, params: Dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        outputs = functional_call(self.model, params, (x.unsqueeze(0).to(self.device),))
        return self.loss(outputs, y.unsqueeze(0))

    def _per_sample_gradient_dict(self, batch: TorchBatch) -> Dict[str, torch.Tensor]:
        return self._per_sample_gradient_function(
            self.params_to_restrict_to, batch.x, batch.y
        )

    def _per_sample_mixed_gradient_dict(
        self, batch: TorchBatch
    ) -> Dict[str, torch.Tensor]:
        return self._per_sample_mixed_gradient_func(
            self.params_to_restrict_to, batch.x, batch.y
        )

    def _matrix_jacobian_product(
        self,
        batch: TorchBatch,
        g: torch.Tensor,
    ) -> torch.Tensor:
        matrix_jacobian_product_func = create_matrix_jacobian_product_function(
            self.model, self.loss, g
        )
        return matrix_jacobian_product_func(
            self.params_to_restrict_to, batch.x, batch.y
        )


GradientProviderFactoryType = Callable[
    [torch.nn.Module, LossType, Optional[Dict[str, torch.nn.Parameter]]],
    TorchPerSampleGradientProvider,
]

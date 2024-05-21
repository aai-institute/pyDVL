from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Union, Type

import torch

from ..functional import create_batch_hvp_function
from ..util import (
    inverse_rank_one_update,
    rank_one_mvp,
    LossType,
    TorchBatch,
)

from .gradient_provider import (
    TorchPerSampleGradientProvider,
    TorchPerSampleAutoGrad,
    GradientProviderFactoryType,
)


class BatchOperation(ABC):
    def __init__(self, regularization: float = 0.0):
        if regularization < 0:
            raise ValueError("regularization must be non-negative")
        self._regularization = regularization

    @property
    @abstractmethod
    def n_parameters(self):
        pass

    @property
    def regularization(self):
        return self._regularization

    @regularization.setter
    def regularization(self, value: float):
        if value < 0:
            raise ValueError("regularization must be non-negative")
        self._regularization = value

    @property
    @abstractmethod
    def device(self):
        pass

    @property
    @abstractmethod
    def dtype(self):
        pass

    @abstractmethod
    def to(self, device: torch.device):
        pass

    @abstractmethod
    def _apply_to_vec(self, batch: TorchBatch, vec: torch.Tensor) -> torch.Tensor:
        pass

    def apply_to_vec(self, batch: TorchBatch, vec: torch.Tensor):
        return self._apply_to_vec(batch.to(self.device), vec.to(self.device))

    def apply_to_mat(self, batch: TorchBatch, mat: torch.Tensor) -> torch.Tensor:
        return torch.func.vmap(
            lambda _x, _y, m: self._apply_to_vec(TorchBatch(_x, _y), m),
            in_dims=(None, None, 0),
            randomness="same",
        )(batch.x, batch.y, mat)


class ModelBasedBatchOperation(BatchOperation, ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: LossType,
        regularization: float = 0.0,
        restrict_to: Optional[Dict[str, torch.nn.Parameter]] = None,
    ):
        super().__init__(regularization)
        if restrict_to is None:
            restrict_to = {
                k: p.detach() for k, p in model.named_parameters() if p.requires_grad
            }
        self.params_to_restrict_to = restrict_to
        self.loss = loss
        self.model = model

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.params_to_restrict_to.values())

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        self.params_to_restrict_to = {
            k: p.detach()
            for k, p in self.model.named_parameters()
            if k in self.params_to_restrict_to
        }
        return self


class HessianBatchOperation(ModelBasedBatchOperation):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: LossType,
        regularization: float = 0.0,
        restrict_to: Optional[Dict[str, torch.nn.Parameter]] = None,
        reverse_only: bool = True,
    ):
        super().__init__(
            model, loss, regularization=regularization, restrict_to=restrict_to
        )
        self._batch_hvp = create_batch_hvp_function(
            model, loss, reverse_only=reverse_only
        )

    def _apply_to_vec(self, batch: TorchBatch, vec: torch.Tensor) -> torch.Tensor:
        return self._batch_hvp(self.params_to_restrict_to, batch.x, batch.y, vec)


class GaussNewtonBatchOperation(ModelBasedBatchOperation):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: LossType,
        regularization: float = 0.0,
        gradient_provider_factory: Union[
            GradientProviderFactoryType,
            Type[TorchPerSampleGradientProvider],
        ] = TorchPerSampleAutoGrad,
        restrict_to: Optional[Dict[str, torch.nn.Parameter]] = None,
    ):
        super().__init__(
            model, loss, regularization=regularization, restrict_to=restrict_to
        )
        self.gradient_provider = gradient_provider_factory(
            model, loss, self.params_to_restrict_to
        )

    def _apply_to_vec(self, batch: TorchBatch, vec: torch.Tensor) -> torch.Tensor:
        flat_grads = self.gradient_provider.per_sample_flat_gradient(batch)
        result = rank_one_mvp(flat_grads, vec)

        if self.regularization > 0.0:
            result += self.regularization * vec

        return result

    def apply_to_mat(self, batch: TorchBatch, mat: torch.Tensor) -> torch.Tensor:
        return self.apply_to_vec(batch, mat)

    def to(self, device: torch.device):
        self.gradient_provider = self.gradient_provider.to(device)
        return super().to(device)


class InverseHarmonicMeanBatchOperation(ModelBasedBatchOperation):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        regularization: float,
        gradient_provider_factory: Union[
            GradientProviderFactoryType,
            Type[TorchPerSampleGradientProvider],
        ] = TorchPerSampleAutoGrad,
        restrict_to: Optional[Dict[str, torch.nn.Parameter]] = None,
    ):
        if regularization <= 0:
            raise ValueError("regularization must be positive")

        super().__init__(
            model, loss, regularization=regularization, restrict_to=restrict_to
        )
        self.regularization = regularization
        self.gradient_provider = gradient_provider_factory(
            model, loss, self.params_to_restrict_to
        )

    def _apply_to_vec(self, batch: TorchBatch, vec: torch.Tensor) -> torch.Tensor:
        grads = self.gradient_provider.per_sample_flat_gradient(batch)
        return (
            inverse_rank_one_update(grads, vec, self.regularization)
            / self.regularization
        )

    def apply_to_mat(self, batch: TorchBatch, mat: torch.Tensor) -> torch.Tensor:
        return self.apply_to_vec(batch, mat)

    def to(self, device: torch.device):
        super().to(device)
        self.gradient_provider.params_to_restrict_to = self.params_to_restrict_to
        return self

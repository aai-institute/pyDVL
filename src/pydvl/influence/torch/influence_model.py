from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import torch
from torch import nn as nn
from torch.utils.data import DataLoader

from .functional import (
    get_hessian,
    matrix_jacobian_product,
    per_sample_gradient,
    per_sample_mixed_derivative,
)
from .torch_differentiable import (
    LowRankProductRepresentation,
    TorchTwiceDifferentiable,
    model_hessian_low_rank,
    solve_arnoldi,
    solve_batch_cg,
    solve_lissa,
)
from .util import flatten_dimensions


class TorchInfluence(ABC):
    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        self.loss = loss
        self.model = model
        self._num_parameters = sum(
            [p.numel() for p in model.parameters() if p.requires_grad]
        )
        self._model_device = next(
            (p.device for p in model.parameters() if p.requires_grad)
        )
        self._model_params = {
            k: p for k, p in self.model.named_parameters() if p.requires_grad
        }
        super().__init__()

    @property
    def num_parameters(self):
        return self._num_parameters

    @property
    def model_device(self):
        return self._model_device

    @property
    def model_params(self):
        return self._model_params

    def _flat_loss_grad(
        self, z: Tuple[torch.Tensor, torch.Tensor], detach: bool = False
    ):
        grads = per_sample_gradient(self.model, self.loss)(self.model_params, *z)
        return flatten_dimensions(
            map(lambda t: t.detach() if detach else t, grads.values()), keep_first_n=0
        )

    def up_weighting(
        self,
        z_test: Tuple[torch.Tensor, torch.Tensor],
        z: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        left = self._flat_loss_grad(z_test, detach=True)
        right = self.factors(z)
        return (left @ right.T).T

    def perturbation(
        self,
        z_test: Tuple[torch.Tensor, torch.Tensor],
        z: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        left = self.factors(z_test)
        right = per_sample_mixed_derivative(self.model, self.loss)(
            self.model_params, *z
        )
        flat_right = flatten_dimensions(right.values(), keep_first_n=2)
        return torch.einsum("ia,jab->ijb", left, flat_right)

    def factors(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        rhs = self._flat_loss_grad(x, detach=True)
        return self._solve_hvp(rhs)

    @abstractmethod
    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    @abstractmethod
    def from_model(
        cls,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_dataloader: DataLoader,
        hessian_perturbation: float,
        **kwargs,
    ):
        pass


class DirectInfluence(TorchInfluence):
    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        hessian: torch.Tensor,
        hessian_perturbation: float,
    ):
        super().__init__(model, loss)
        self.hessian_perturbation = hessian_perturbation
        self.hessian = hessian
        self.perturbed_matrix = hessian + hessian_perturbation * torch.eye(
            self.num_parameters, device=self.model_device
        )

    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(self.perturbed_matrix, rhs.T).T

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_dataloader: DataLoader,
        hessian_perturbation: float,
        use_average_hessian: bool = True,
    ):
        hessian = get_hessian(
            model,
            loss,
            train_dataloader,
            use_hessian_avg=use_average_hessian,
        )
        return cls(model, loss, hessian, hessian_perturbation)


class BatchCgInfluence(TorchInfluence):
    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_dataloader: DataLoader,
        hessian_perturbation: float,
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
        self.hessian_perturbation = hessian_perturbation
        self.train_dataloader = train_dataloader

    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        # TODO directly implement the method here and remove call to obsolete function
        x, _ = solve_batch_cg(
            TorchTwiceDifferentiable(self.model, self.loss),
            self.train_dataloader,
            rhs,
            self.hessian_perturbation,
            x0=self.x0,
            rtol=self.rtol,
            atol=self.atol,
            maxiter=self.maxiter,
        )
        return x

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_dataloader: DataLoader,
        hessian_perturbation: float,
        x0: Optional[torch.Tensor] = None,
        rtol: float = 1e-7,
        atol: float = 1e-7,
        maxiter: Optional[int] = None,
        progress: bool = False,
    ):
        return cls(
            model,
            loss,
            train_dataloader,
            hessian_perturbation,
            x0,
            rtol,
            atol,
            maxiter,
            progress,
        )


class LissaInfluence(TorchInfluence):
    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_dataloader: DataLoader,
        hessian_perturbation: float,
        maxiter: int = 1000,
        dampen: float = 0.0,
        scale: float = 10.0,
        h0: Optional[torch.Tensor] = None,
        rtol: float = 1e-4,
        progress: bool = False,
    ):
        super().__init__(model, loss)
        self.maxiter = maxiter
        self.hessian_perturbation = hessian_perturbation
        self.progress = progress
        self.rtol = rtol
        self.h0 = h0
        self.scale = scale
        self.dampen = dampen
        self.train_dataloader = train_dataloader

    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        # TODO directly implement the method here and remove call to obsolete function
        x, _ = solve_lissa(
            TorchTwiceDifferentiable(self.model, self.loss),
            self.train_dataloader,
            rhs,
            self.hessian_perturbation,
            maxiter=self.maxiter,
            dampen=self.dampen,
            scale=self.scale,
            h0=self.h0,
            rtol=self.rtol,
            progress=self.progress,
        )
        return x

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_dataloader: DataLoader,
        hessian_perturbation: float,
        maxiter: int = 1000,
        dampen: float = 0.0,
        scale: float = 10.0,
        h0: Optional[torch.Tensor] = None,
        rtol: float = 1e-4,
        progress: bool = False,
    ):
        return cls(
            model,
            loss,
            train_dataloader,
            hessian_perturbation,
            maxiter,
            dampen,
            scale,
            h0,
            rtol,
            progress,
        )


class ArnoldiInfluence(TorchInfluence):
    def __init__(
        self, model, loss, low_rank_representation: LowRankProductRepresentation
    ):
        self.low_rank_representation = low_rank_representation
        super().__init__(model, loss)

    def up_weighting(
        self, x: Tuple[torch.Tensor, torch.Tensor], y: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        mjp = matrix_jacobian_product(
            self.model, self.loss, self.low_rank_representation.projections
        )
        left = mjp(self.model_params, *x)
        right = torch.diag_embed(1.0 / self.low_rank_representation.eigen_vals) @ mjp(
            self.model_params, *y
        )
        return torch.einsum("ij, kj -> ik", left, right)

    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        x, _ = solve_arnoldi(
            TorchTwiceDifferentiable(self.model, self.loss),
            None,
            rhs,
            low_rank_representation=self.low_rank_representation,
        )
        return x

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_dataloader: DataLoader,
        hessian_perturbation: float,
        rank_estimate: int = 10,
        krylov_dimension: Optional[int] = None,
        tol: float = 1e-6,
        max_iter: Optional[int] = None,
        eigen_computation_on_gpu: bool = False,
    ):
        low_rank_representation = model_hessian_low_rank(
            TorchTwiceDifferentiable(model, loss),
            train_dataloader,
            hessian_perturbation,
            rank_estimate,
            krylov_dimension,
            tol,
            max_iter,
            eigen_computation_on_gpu,
        )
        return cls(model, loss, low_rank_representation)

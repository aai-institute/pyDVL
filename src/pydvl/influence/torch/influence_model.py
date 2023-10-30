from abc import ABC, abstractmethod
from math import prod
from typing import Callable, Optional, Tuple

import torch
from torch import nn as nn
from torch.utils.data import DataLoader

from ..inversion import InfluenceRegistry, InversionMethod
from ..twice_differentiable import Influence, InfluenceType, TensorType
from .functional import (
    get_hessian,
    get_hvp_function,
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


class TorchInfluence(Influence[torch.Tensor], ABC):
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
            k: p.detach() for k, p in self.model.named_parameters() if p.requires_grad
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

    def _flat_loss_grad(self, z: Tuple[torch.Tensor, torch.Tensor]):
        grads = per_sample_gradient(self.model, self.loss)(self.model_params, *z)
        shape = (z[0].shape[0], -1)
        return flatten_dimensions(grads.values(), shape=shape)

    def _flat_loss_mixed_grad(self, z: Tuple[torch.Tensor, torch.Tensor]):
        mixed_grads = per_sample_mixed_derivative(self.model, self.loss)(
            self.model_params, *z
        )
        shape = (z[0].shape[0], prod(z[0].shape[1:]), -1)
        return flatten_dimensions(mixed_grads.values(), shape=shape)

    def values(
        self,
        z_test: Tuple[torch.Tensor, torch.Tensor],
        z: Tuple[torch.Tensor, torch.Tensor],
        influence_type: InfluenceType,
    ) -> torch.Tensor:
        if influence_type is InfluenceType.Up:
            return self.up_weighting(self.factors(z_test), z)
        return self.perturbation(self.factors(z_test), z)

    def up_weighting(
        self, z_test_factors: torch.Tensor, z: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return z_test_factors @ self._flat_loss_grad(z).T

    def perturbation(
        self,
        z_test_factors: torch.Tensor,
        z: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return torch.einsum(
            "ia,jba->ijb", z_test_factors, self._flat_loss_mixed_grad(z)
        )

    def factors(self, z: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = z
        x, y = x.to(self.model_device), y.to(self.model_device)
        return self._solve_hvp(self._flat_loss_grad((x, y)))

    @abstractmethod
    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        pass


class DirectInfluence(TorchInfluence):
    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        hessian_regularization: float,
        hessian: torch.Tensor = None,
        data_loader: DataLoader = None,
    ):
        if hessian is None and data_loader is None:
            raise ValueError(
                f"Either provide a pre-computed hessian or a data_loader to compute the hessian"
            )

        super().__init__(model, loss)
        self.hessian_perturbation = hessian_regularization
        self.hessian = (
            hessian if hessian is not None else get_hessian(model, loss, data_loader)
        )

    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(
            self.hessian
            + self.hessian_perturbation
            * torch.eye(self.num_parameters, device=self.model_device),
            rhs.T,
        ).T


class BatchCgInfluence(TorchInfluence):
    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_dataloader: DataLoader,
        hessian_regularization: float,
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
        self.hessian_regularization = hessian_regularization
        self.train_dataloader = train_dataloader

    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        # TODO directly implement the method here and remove call to obsolete function
        x, _ = solve_batch_cg(
            TorchTwiceDifferentiable(self.model, self.loss),
            self.train_dataloader,
            rhs,
            self.hessian_regularization,
            x0=self.x0,
            rtol=self.rtol,
            atol=self.atol,
            maxiter=self.maxiter,
        )
        return x


class LissaInfluence(TorchInfluence):
    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_dataloader: DataLoader,
        hessian_regularization: float,
        maxiter: int = 1000,
        dampen: float = 0.0,
        scale: float = 10.0,
        h0: Optional[torch.Tensor] = None,
        rtol: float = 1e-4,
        progress: bool = False,
    ):
        super().__init__(model, loss)
        self.maxiter = maxiter
        self.hessian_regularization = hessian_regularization
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
            self.hessian_regularization,
            maxiter=self.maxiter,
            dampen=self.dampen,
            scale=self.scale,
            h0=self.h0,
            rtol=self.rtol,
            progress=self.progress,
        )
        return x


class ArnoldiInfluence(TorchInfluence):
    def __init__(
        self,
        model,
        loss,
        low_rank_representation: Optional[LowRankProductRepresentation] = None,
        data_loader: DataLoader = None,
        hessian_regularization: float = 0.0,
        rank_estimate: int = 10,
        krylov_dimension: Optional[int] = None,
        tol: float = 1e-6,
        max_iter: Optional[int] = None,
        eigen_computation_on_gpu: bool = False,
    ):
        if low_rank_representation is None and data_loader is None:
            raise ValueError(
                f"Either provide a pre-computed hessian or a data_loader to compute the hessian"
            )

        if low_rank_representation is None:
            low_rank_representation = model_hessian_low_rank(
                model,
                loss,
                data_loader,
                hessian_perturbation=hessian_regularization,
                rank_estimate=rank_estimate,
                krylov_dimension=krylov_dimension,
                tol=tol,
                max_iter=max_iter,
                eigen_computation_on_gpu=eigen_computation_on_gpu,
            )

        self.low_rank_representation = low_rank_representation

        super().__init__(model, loss)

    def values(
        self,
        z_test: Tuple[torch.Tensor, torch.Tensor],
        z: Tuple[torch.Tensor, torch.Tensor],
        influence_type: InfluenceType,
    ) -> torch.Tensor:

        if influence_type is InfluenceType.Up:
            mjp = matrix_jacobian_product(
                self.model, self.loss, self.low_rank_representation.projections.T
            )
            left = mjp(self.model_params, *z_test)
            right = torch.diag_embed(
                1.0 / self.low_rank_representation.eigen_vals
            ) @ mjp(self.model_params, *z)
            return torch.einsum("ij, ik -> jk", left, right)

        return self.perturbation(self.factors(z_test), z)

    def _solve_hvp(self, rhs: torch.Tensor) -> torch.Tensor:
        x, _ = solve_arnoldi(
            TorchTwiceDifferentiable(self.model, self.loss),
            None,
            rhs,
            low_rank_representation=self.low_rank_representation,
        )
        return x


@InfluenceRegistry.register(TorchTwiceDifferentiable, InversionMethod.Direct)
def direct_factory(
    twice_differentiable: TorchTwiceDifferentiable,
    data_loader: DataLoader,
    hessian_regularization: float,
    **kwargs,
):
    return DirectInfluence(
        twice_differentiable.model,
        twice_differentiable.loss,
        data_loader=data_loader,
        hessian_regularization=hessian_regularization,
        **kwargs,
    )


@InfluenceRegistry.register(TorchTwiceDifferentiable, InversionMethod.Cg)
def cg_factory(
    twice_differentiable: TorchTwiceDifferentiable,
    data_loader: DataLoader,
    hessian_regularization: float,
    **kwargs,
):
    return BatchCgInfluence(
        twice_differentiable.model,
        twice_differentiable.loss,
        train_dataloader=data_loader,
        hessian_regularization=hessian_regularization,
        **kwargs,
    )


@InfluenceRegistry.register(TorchTwiceDifferentiable, InversionMethod.Lissa)
def lissa_factory(
    twice_differentiable: TorchTwiceDifferentiable,
    data_loader: DataLoader,
    hessian_regularization: float,
    **kwargs,
):
    return LissaInfluence(
        twice_differentiable.model,
        twice_differentiable.loss,
        train_dataloader=data_loader,
        hessian_regularization=hessian_regularization,
        **kwargs,
    )


@InfluenceRegistry.register(TorchTwiceDifferentiable, InversionMethod.Arnoldi)
def arnoldi_factory(
    twice_differentiable: TorchTwiceDifferentiable,
    data_loader: DataLoader,
    hessian_regularization: float,
    **kwargs,
):
    return ArnoldiInfluence(
        twice_differentiable.model,
        twice_differentiable.loss,
        data_loader=data_loader,
        hessian_regularization=hessian_regularization,
        **kwargs,
    )

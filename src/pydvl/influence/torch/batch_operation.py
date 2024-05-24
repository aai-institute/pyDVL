r"""
This module contains abstractions and implementations for operations carried out on a
batch $b$. These operations are of the form

$$ m(b) \cdot v$$,

where $m(b)$ is a matrix defined by the data in the batch and $v$ is a vector or matrix.
These batch operations can be used to conveniently build aggregations or recursions
over sequence of batches, e.g. an average of the form

$$ \frac{1}{|B|} \sum_{b in B}m(b)\cdot v$$,

which is useful in the case that keeping $B$ in memory is not feasible.

"""
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Type, TypeVar, Union

import torch

from .base import (
    GradientProviderFactoryType,
    TorchAutoGrad,
    TorchBatch,
    TorchGradientProvider,
)
from .functional import create_batch_hvp_function
from .util import LossType, inverse_rank_one_update, rank_one_mvp


class BatchOperation(ABC):
    r"""
    Abstract base class to implement operations of the form

    $$ m(b) \cdot v $$

    where $m(b)$ is a matrix defined by the data in the batch and $v$ is a vector
    or matrix.
    """

    @property
    @abstractmethod
    def input_size(self):
        pass

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
        """
        Applies the batch operation to a single vector.
        Args:
            batch: Batch of data for computation
            vec: A single vector consistent to the operation, i.e. it's length
                must be equal to the property `input_size`.

        Returns:
            A single vector after applying the batch operation
        """
        return self._apply_to_vec(batch.to(self.device), vec.to(self.device))

    def apply_to_mat(self, batch: TorchBatch, mat: torch.Tensor) -> torch.Tensor:
        """
        Applies the batch operation to a matrix.
        Args:
            batch: Batch of data for computation
            mat: A matrix to apply the batch operation to. The last dimension is
                assumed to be consistent to the operation, i.e. it must equal
                to the property `input_size`.

        Returns:
            A matrix of shape $(N, \text{input_size})$, given the shape of mat is
                $(N, \text{input_size})$

        """
        return torch.func.vmap(
            lambda _x, _y, m: self._apply_to_vec(TorchBatch(_x, _y), m),
            in_dims=(None, None, 0),
            randomness="same",
        )(batch.x, batch.y, mat)


class ModelBasedBatchOperation(BatchOperation, ABC):
    r"""
    Abstract base class to implement operations of the form

    $$ m(\text{model}, b) \cdot v $$

    where model is a [torch.nn.Module][torch.nn.Module].

    """

    def __init__(
        self,
        model: torch.nn.Module,
        restrict_to: Optional[Dict[str, torch.nn.Parameter]] = None,
    ):
        if restrict_to is None:
            restrict_to = {
                k: p.detach() for k, p in model.named_parameters() if p.requires_grad
            }
        self.params_to_restrict_to = restrict_to
        self.model = model

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def input_size(self):
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
    r"""
    Given a model and loss function computes the Hessian vector or matrix product
    with respect to the model parameters, i.e.

    \begin{align*}
        &\nabla^2_{\theta} L(b;\theta) \cdot v \\\
        &L(b;\theta) = \left( \frac{1}{|b|} \sum_{(x,y) \in b}
        \text{loss}(\text{model}(x; \theta), y)\right),
    \end{align*}

    where model is a [torch.nn.Module][torch.nn.Module] and $v$ is a vector or matrix.

    Args:
        model: The model.
        loss: The loss function.
        restrict_to: The parameters to restrict the second order differentiation to,
            i.e. the corresponding sub-matrix of the Hessian. If None, the full Hessian
            is used. Make sure the input matches the corrct dimension, i.e. the
            last dimension must be equal to the property `input_size`.
        reverse_only: If True only the reverse mode is used in the autograd computation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: LossType,
        restrict_to: Optional[Dict[str, torch.nn.Parameter]] = None,
        reverse_only: bool = True,
    ):
        super().__init__(model, restrict_to=restrict_to)
        self._batch_hvp = create_batch_hvp_function(
            model, loss, reverse_only=reverse_only
        )

    def _apply_to_vec(self, batch: TorchBatch, vec: torch.Tensor) -> torch.Tensor:
        return self._batch_hvp(self.params_to_restrict_to, batch.x, batch.y, vec)


class GaussNewtonBatchOperation(ModelBasedBatchOperation):
    r"""
    Given a model and loss function computes the Gauss-Newton vector or matrix product
    with respect to the model parameters, i.e.

    \begin{align*}
        G(\text{model}, \text{loss}, b, \theta) &\cdot v, \\\
        G(\text{model}, \text{loss}, b, \theta) &=
        \frac{1}{|b|}\sum_{(x, y) \in b}\nabla_{\theta}\ell (x,y; \theta)
            \nabla_{\theta}\ell (x,y; \theta)^t, \\\
        \ell(x,y; \theta) &= \text{loss}(\text{model}(x; \theta), y)
    \end{align*}

    where model is a [torch.nn.Module][torch.nn.Module] and $v$ is a vector or matrix.

    Args:
        model: The model.
        loss: The loss function.
        gradient_provider_factory: An optional factory to create an object of type
            [TorchGradientProvider][pydvl.influence.torch.base.TorchGradientProvider],
            depending on the model, loss and optional parameters to restrict to.
            If not provided, the implementation
            [TorchAutograd][pydvl.influence.torch.base.TorchAutograd] is used.
        restrict_to: The parameters to restrict the differentiation to,
            i.e. the corresponding sub-matrix of the Jacobian. If None, the full
            Jacobian is used. Make sure the input matches the corrct dimension, i.e. the
            last dimension must be equal to the property `input_size`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: LossType,
        gradient_provider_factory: Union[
            GradientProviderFactoryType,
            Type[TorchGradientProvider],
        ] = TorchAutoGrad,
        restrict_to: Optional[Dict[str, torch.nn.Parameter]] = None,
    ):
        super().__init__(model, restrict_to=restrict_to)
        self.gradient_provider = gradient_provider_factory(
            model, loss, self.params_to_restrict_to
        )

    def _apply_to_vec(self, batch: TorchBatch, vec: torch.Tensor) -> torch.Tensor:
        flat_grads = self.gradient_provider.flat_grads(batch)
        result = rank_one_mvp(flat_grads, vec)
        return result

    def apply_to_mat(self, batch: TorchBatch, mat: torch.Tensor) -> torch.Tensor:
        """
        Applies the batch operation to a matrix.
        Args:
            batch: Batch of data for computation
            mat: A matrix to apply the batch operation to. The last dimension is
                assumed to be consistent to the operation, i.e. it must equal
                to the property `input_size`.

        Returns:
            A matrix of shape $(N, \text{input_size})$, given the shape of mat is
                $(N, \text{input_size})$

        """
        return self.apply_to_vec(batch, mat)

    def to(self, device: torch.device):
        self.gradient_provider = self.gradient_provider.to(device)
        return super().to(device)


class InverseHarmonicMeanBatchOperation(ModelBasedBatchOperation):
    r"""
    Given a model and loss function computes an approximation of the inverse
    Gauss-Newton vector or matrix product. Viewing the damped Gauss-newton matrix

    \begin{align*}
        G_{\lambda}(\text{model}, \text{loss}, b, \theta) &=
        \frac{1}{|b|}\sum_{(x, y) \in b}\nabla_{\theta}\ell (x,y; \theta)
            \nabla_{\theta}\ell (x,y; \theta)^t + \lambda \operatorname{I}, \\\
        \ell(x,y; \theta) &= \text{loss}(\text{model}(x; \theta), y)
    \end{align*}

    as an arithmetic mean of the rank-$1$ updates, this operation replaces it with
    the harmonic mean of the rank-$1$ updates, i.e.

    $$ \tilde{G}_{\lambda}(\text{model}, \text{loss}, b, \theta) =
        \left(n \sum_{(x, y) \in b}  \left( \nabla_{\theta}\ell (x,y; \theta)
            \nabla_{\theta}\ell (x,y; \theta)^t + \lambda \operatorname{I}\right)^{-1}
            \right)^{-1}$$

    and computes

    $$ \tilde{G}_{\lambda}^{-1}(\text{model}, \text{loss}, b, \theta)
    \cdot v.$$

    where model is a [torch.nn.Module][torch.nn.Module] and $v$ is a vector or matrix.
    In other words, it switches the order of summation and inversion, which resolves
    to the `inverse harmonic mean` of the rank-$1$ updates.

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
        restrict_to: The parameters to restrict the differentiation to,
            i.e. the corresponding sub-matrix of the Jacobian. If None, the full
            Jacobian is used. Make sure the input matches the corrct dimension, i.e. the
            last dimension must be equal to the property `input_size`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        regularization: float,
        gradient_provider_factory: Union[
            GradientProviderFactoryType,
            Type[TorchGradientProvider],
        ] = TorchAutoGrad,
        restrict_to: Optional[Dict[str, torch.nn.Parameter]] = None,
    ):
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        self.regularization = regularization

        super().__init__(model, restrict_to=restrict_to)
        self.gradient_provider = gradient_provider_factory(
            model, loss, self.params_to_restrict_to
        )

    @property
    def regularization(self):
        return self._regularization

    @regularization.setter
    def regularization(self, value: float):
        if value <= 0:
            raise ValueError("regularization must be positive")
        self._regularization = value

    def _apply_to_vec(self, batch: TorchBatch, vec: torch.Tensor) -> torch.Tensor:
        grads = self.gradient_provider.flat_grads(batch)
        return (
            inverse_rank_one_update(grads, vec, self.regularization)
            / self.regularization
        )

    def apply_to_mat(self, batch: TorchBatch, mat: torch.Tensor) -> torch.Tensor:
        """
        Applies the batch operation to a matrix.
        Args:
            batch: Batch of data for computation
            mat: A matrix to apply the batch operation to. The last dimension is
                assumed to be consistent to the operation, i.e. it must equal
                to the property `input_size`.

        Returns:
            A matrix of shape $(N, \text{input_size})$, given the shape of mat is
                $(N, \text{input_size})$

        """
        return self.apply_to_vec(batch, mat)

    def to(self, device: torch.device):
        super().to(device)
        self.gradient_provider.params_to_restrict_to = self.params_to_restrict_to
        return self


BatchOperationType = TypeVar("BatchOperationType", bound=BatchOperation)

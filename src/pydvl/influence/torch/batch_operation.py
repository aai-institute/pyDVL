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
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch

from .base import (
    GradientProviderFactoryType,
    TorchAutoGrad,
    TorchBatch,
    TorchGradientProvider,
)
from .functional import create_batch_hvp_function, create_batch_loss_function, hvp
from .util import (
    LossType,
    generate_inverse_rank_one_updates,
    generate_rank_one_mvp,
    inverse_rank_one_update,
    rank_one_mvp,
)


class _ModelBasedBatchOperation(ABC):
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
    def input_dict_structure(self) -> Dict[str, Tuple[int, ...]]:
        return {k: p.shape for k, p in self.params_to_restrict_to.items()}

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

    def apply_to_tensor_dict(
        self, batch: TorchBatch, mat_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        if mat_dict.keys() != self.params_to_restrict_to.keys():
            raise ValueError(
                "The keys of the matrix dictionary must match the keys of the "
                "parameters to restrict to."
            )

        return self._apply_to_tensor_dict(
            batch, {k: v.to(self.device) for k, v in mat_dict.items()}
        )

    def _has_batch_dim(self, tensor_dict: Dict[str, torch.Tensor]):
        batch_dim_flags = [
            tensor_dict[key].shape == val.shape
            for key, val in self.params_to_restrict_to.items()
        ]
        if len(set(batch_dim_flags)) == 2:
            raise ValueError("Existence of batch dim must be consistent")
        return not all(batch_dim_flags)

    def _add_batch_dim(self, vec_dict: Dict[str, torch.Tensor]):
        result = {}
        for key, value in self.params_to_restrict_to.items():
            if value.shape == vec_dict[key].shape:
                result[key] = vec_dict[key].unsqueeze(0)
            else:
                result[key] = vec_dict[key]
        return result

    @abstractmethod
    def _apply_to_tensor_dict(
        self, batch: TorchBatch, mat_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
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


class HessianBatchOperation(_ModelBasedBatchOperation):
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
        self.loss = loss
        self.reverse_only = reverse_only

    def _apply_to_vec(self, batch: TorchBatch, vec: torch.Tensor) -> torch.Tensor:
        return self._batch_hvp(self.params_to_restrict_to, batch.x, batch.y, vec)

    def _apply_to_tensor_dict(
        self, batch: TorchBatch, mat_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        func = self._create_seq_func(*batch)

        if self._has_batch_dim(mat_dict):
            func = torch.func.vmap(
                func, in_dims=tuple((0 for _ in self.params_to_restrict_to))
            )

        result: Dict[str, torch.Tensor] = func(*mat_dict.values())
        return result

    def _create_seq_func(self, x: torch.Tensor, y: torch.Tensor):
        def seq_func(*vec: torch.Tensor) -> Dict[str, torch.Tensor]:
            return hvp(
                lambda p: create_batch_loss_function(self.model, self.loss)(p, x, y),
                self.params_to_restrict_to,
                dict(zip(self.params_to_restrict_to.keys(), vec)),
                reverse_only=self.reverse_only,
            )

        return seq_func


class GaussNewtonBatchOperation(_ModelBasedBatchOperation):
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

    def _apply_to_tensor_dict(
        self, batch: TorchBatch, vec_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        vec_values = list(self._add_batch_dim(vec_dict).values())
        grads_dict = self.gradient_provider.grads(batch)
        grads_values = list(self._add_batch_dim(grads_dict).values())
        gen_result = generate_rank_one_mvp(grads_values, vec_values)
        return dict(zip(vec_dict.keys(), gen_result))

    def _apply_to_vec(self, batch: TorchBatch, vec: torch.Tensor) -> torch.Tensor:
        flat_grads = self.gradient_provider.flat_grads(batch)
        return rank_one_mvp(flat_grads, vec)

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


class InverseHarmonicMeanBatchOperation(_ModelBasedBatchOperation):
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
        if vec.ndim == 1:
            input_vec = vec.unsqueeze(0)
        else:
            input_vec = vec
        return inverse_rank_one_update(grads, input_vec, self.regularization)

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

    def _apply_to_tensor_dict(
        self, batch: TorchBatch, vec_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        vec_values = list(self._add_batch_dim(vec_dict).values())
        grads_dict = self.gradient_provider.grads(batch)
        grads_values = list(self._add_batch_dim(grads_dict).values())
        gen_result = generate_inverse_rank_one_updates(
            grads_values, vec_values, self.regularization
        )
        return dict(zip(vec_dict.keys(), gen_result))


BatchOperationType = TypeVar("BatchOperationType", bound=_ModelBasedBatchOperation)


class _TensorDictAveraging(ABC):
    @abstractmethod
    def __call__(self, tensor_dicts: Generator[Dict[str, torch.Tensor], None, None]):
        pass


_TensorDictAveragingType = TypeVar(
    "_TensorDictAveragingType", bound=_TensorDictAveraging
)


class _TensorAveraging(Generic[_TensorDictAveragingType], ABC):
    @abstractmethod
    def __call__(self, tensors: Generator[torch.Tensor, None, None]):
        pass

    @abstractmethod
    def as_dict_averaging(self) -> _TensorDictAveraging:
        pass


TensorAveragingType = TypeVar("TensorAveragingType", bound=_TensorAveraging)


class _TensorDictChunkAveraging(_TensorDictAveraging):
    def __call__(self, tensor_dicts: Generator[Dict[str, torch.Tensor], None, None]):
        result = next(tensor_dicts)
        n_chunks = 1.0
        for tensor_dict in tensor_dicts:
            for key, tensor in tensor_dict.items():
                result[key] += tensor
            n_chunks += 1.0
        return {k: t / n_chunks for k, t in result.items()}


class ChunkAveraging(_TensorAveraging[_TensorDictChunkAveraging]):
    """
    Averages tensors, provided by a generator, and normalizes by the number
    of tensors.
    """

    def __call__(self, tensors: Generator[torch.Tensor, None, None]):
        result = next(tensors)
        n_chunks = 1.0
        for tensor in tensors:
            result += tensor
            n_chunks += 1.0
        return result / n_chunks

    def as_dict_averaging(self) -> _TensorDictChunkAveraging:
        return _TensorDictChunkAveraging()


class _TensorDictPointAveraging(_TensorDictAveraging):
    def __init__(self, batch_dim: int = 0):
        self.batch_dim = batch_dim

    def __call__(self, tensor_dicts: Generator[Dict[str, torch.Tensor], None, None]):
        result = next(tensor_dicts)
        n_points = next(iter(result.values())).shape[self.batch_dim]
        for tensor_dict in tensor_dicts:
            n_points_in_batch = next(iter(tensor_dict.values())).shape[self.batch_dim]
            for key, tensor in tensor_dict.items():
                result[key] += n_points_in_batch * tensor
            n_points += n_points_in_batch
        return {k: t / float(n_points) for k, t in result.items()}


class PointAveraging(_TensorAveraging[_TensorDictPointAveraging]):
    """
    Averages tensors provided by a generator. The averaging is weighted by
    the number of points in each tensor and the final result is normalized by the
    number of total points.

    Args:
        batch_dim: Dimension to extract the number of points for the weighting.

    """

    def __init__(self, batch_dim: int = 0):
        self.batch_dim = batch_dim

    def __call__(self, tensors: Generator[torch.Tensor, None, None]):
        result = next(tensors)
        n_points = result.shape[self.batch_dim]
        for tensor in tensors:
            n_points_in_batch = tensor.shape[self.batch_dim]
            result += n_points_in_batch * tensor
            n_points += n_points_in_batch
        return result / float(n_points)

    def as_dict_averaging(self) -> _TensorDictPointAveraging:
        return _TensorDictPointAveraging(self.batch_dim)

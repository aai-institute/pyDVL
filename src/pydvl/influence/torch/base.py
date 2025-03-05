from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import torch
from torch.func import functional_call
from torch.utils.data import DataLoader

from ..base_influence_function_model import ComposableInfluence
from ..types import (
    Batch,
    BilinearForm,
    BlockMapper,
    GradientProvider,
    Operator,
    OperatorGradientComposition,
)
from .util import (
    BlockMode,
    LossType,
    ModelInfoMixin,
    ModelParameterDictBuilder,
    align_structure,
    flatten_dimensions,
)

if TYPE_CHECKING:
    from .operator import LowRankOperator


@dataclass(frozen=True)
class TorchBatch(Batch):
    """
    A convenience class for handling batches of data. Validates the alignment
    of the first dimension (batch dimension) of the input and target tensor

    Attributes:
        x: The input tensor that contains features or data points.
        y: The target tensor that contains labels corresponding to the inputs.

    """

    x: torch.Tensor
    y: torch.Tensor

    def __iter__(self):
        return iter((self.x, self.y))

    def __post_init__(self):
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"The first dimension of x and y must be the same, "
                f"got {self.x.shape[0]} and {self.y.shape[0]}"
            )

    def __len__(self):
        return self.x.shape[0]

    def to(self, device: torch.device):
        return TorchBatch(self.x.to(device), self.y.to(device))


class TorchGradientProvider(GradientProvider[TorchBatch, torch.Tensor]):
    r"""
    Computes per-sample gradients of a function defined by
    a [torch.nn.Module][torch.nn.Module] and a loss function using
    [torch.func][torch.func].

    Consider a function

    $$ \ell: \mathbb{R}^{d_1} \times \mathbb{R}^{d_2} \times \mathbb{R}^{n}
        \times \mathbb{R}^{n}, \quad \ell(\omega_1, \omega_2, x, y) =
        \operatorname{loss}(f(\omega_1, \omega_2; x), y), $$

    e.g. a two layer neural network $f$ with a loss function. This object
    computes the expressions:

    $$ \nabla_{\omega_{i}}\ell(\omega_1, \omega_2, x, y),
       \nabla_{\omega_{i}}\nabla_{x}\ell(\omega_1, \omega_2, x, y),
       \nabla_{\omega}\ell(\omega_1, \omega_2, x, y) \cdot v. $$

    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: LossType,
        restrict_to: Optional[Dict[str, torch.nn.Parameter]],
    ):
        self.model = model
        self.loss = loss

        if restrict_to is None:
            restrict_to = ModelParameterDictBuilder(model).build_from_block_mode(
                BlockMode.FULL
            )

        self.params_to_restrict_to = restrict_to

    def _compute_loss(
        self, params: Dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        outputs = functional_call(self.model, params, (x.unsqueeze(0).to(self.device),))
        return self.loss(outputs, y.unsqueeze(0))

    def _grads(self, batch: TorchBatch) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = torch.vmap(
            torch.func.grad(self._compute_loss), in_dims=(None, 0, 0)
        )(self.params_to_restrict_to, batch.x, batch.y)
        return result

    def _mixed_grads(self, batch: TorchBatch) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = torch.vmap(
            torch.func.jacrev(torch.func.grad(self._compute_loss, argnums=1)),
            in_dims=(None, 0, 0),
        )(self.params_to_restrict_to, batch.x, batch.y)
        return result

    def _jacobian_prod(
        self,
        batch: TorchBatch,
        g: torch.Tensor,
    ) -> torch.Tensor:
        def single_jvp(
            _g: torch.Tensor,
        ):
            return torch.func.jvp(
                lambda p: torch.vmap(self._compute_loss, in_dims=(None, 0, 0))(
                    p, *batch
                ),
                (self.params_to_restrict_to,),
                (align_structure(self.params_to_restrict_to, _g),),
            )[1]

        return torch.func.vmap(single_jvp)(g)

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

    @staticmethod
    def _detach_dict(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: g.detach() if g.requires_grad else g for k, g in tensor_dict.items()}

    def grads(self, batch: TorchBatch) -> Dict[str, torch.Tensor]:
        r"""
        Computes and returns a dictionary mapping parameter names to their respective
        per-sample gradients. Given the example in the class docstring, this means

        $$ \text{result}[\omega_i] = \nabla_{\omega_{i}}\ell(\omega_1, \omega_2,
            \text{batch.x}, \text{batch.y}), $$

        where the first dimension of the resulting tensors is always considered to be
        the batch dimension, so the shape of the resulting tensors are $(N, d_i)$,
        where $N$ is the number of samples in the batch.

        Args:
            batch: The batch of data for which to compute gradients.

        Returns:
            A dictionary where keys are gradient identifiers and values are the
                gradients computed per sample.
        """
        gradient_dict = self._grads(batch.to(self.device))
        return self._detach_dict(gradient_dict)

    def mixed_grads(self, batch: TorchBatch) -> Dict[str, torch.Tensor]:
        r"""
        Computes and returns a dictionary mapping gradient names to their respective
        per-sample mixed gradients. In this context, mixed gradients refer to computing
        gradients with respect to the instance definition in addition to
        compute derivatives with respect to the input batch.
        Given the example in the class docstring, this means

        $$ \text{result}[\omega_i] = \nabla_{\omega_{i}}\nabla_{x}\ell(\omega_1,
            \omega_2, \text{batch.x}, \text{batch.y}), $$

        where the first dimension of the resulting tensors is always considered to be
        the batch dimension and the last to be the non-batch input related derivatives.
        So the shape of the resulting tensors are $(N, n, d_i)$,
        where $N$ is the number of samples in the batch.

        Args:
            batch: The batch of data for which to compute mixed gradients.

        Returns:
            A dictionary where keys are gradient identifiers and values are the
                mixed gradients computed per sample.
        """
        gradient_dict = self._mixed_grads(batch.to(self.device))
        return self._detach_dict(gradient_dict)

    def jacobian_prod(
        self,
        batch: TorchBatch,
        g: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Computes the matrix-Jacobian product for the provided batch and input tensor.
        Given the example in the class docstring, this means

        $$ (\nabla_{\omega_{1}}\ell(\omega_1, \omega_2,
            \text{batch.x}, \text{batch.y}),
            \nabla_{\omega_{2}}\ell(\omega_1, \omega_2,
            \text{batch.x}, \text{batch.y})) \cdot g^T$$

        where g must be a tensor of shape $(K, d_1+d_2)$, so the resulting tensor
        is of shape $(N, K)$.

        Args:
            batch: The batch of data for which to compute the Jacobian.
            g: The tensor to be used in the matrix-Jacobian product
                calculation.

        Returns:
            The resulting tensor from the matrix-Jacobian product computation.
        """
        result = self._jacobian_prod(batch.to(self.device), g.to(self.device))
        if result.requires_grad:
            result = result.detach()
        return result

    def flat_grads(self, batch: TorchBatch) -> torch.Tensor:
        return flatten_dimensions(
            self.grads(batch).values(), shape=(batch.x.shape[0], -1)
        )

    def flat_mixed_grads(self, batch: TorchBatch) -> torch.Tensor:
        shape = (*batch.x.shape, -1)
        return flatten_dimensions(self.mixed_grads(batch).values(), shape=shape)


class OperatorBilinearForm(
    BilinearForm[torch.Tensor, TorchBatch, TorchGradientProvider],
):
    r"""
    Base class for bi-linear forms based on an instance of
    [TensorOperator][pydvl.influence.torch.base.TensorOperator]. This means
    it computes weighted inner products of the form:

    $$ \langle \operatorname{Op}(x), y \rangle $$

    Args:
        operator: The operator to compute the inner product with.
    """

    def __init__(self, operator: TensorOperator):
        self.operator = operator

    def inner_prod(
        self, left: torch.Tensor, right: Optional[torch.Tensor]
    ) -> torch.Tensor:
        r"""Computes the weighted inner product of two vectors, i.e.

        $$ \langle \operatorname{Op}(\text{left}), \text{right} \rangle $$

        Args:
            left: The first tensor in the inner product computation.
            right: The second tensor, optional; if not provided, the inner product will
                use `left` tensor for both arguments.

        Returns:
            A tensor representing the inner product.
        """
        if right is None:
            right = left
        if left.shape[0] <= right.shape[0]:
            return self._inner_product(left, right)
        return self._inner_product(right, left).T

    def _inner_product(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_result = self.operator.apply(left)
        return torch.einsum("ia,j...a->ij...", left_result, right)


class DictBilinearForm(OperatorBilinearForm):
    r"""Base class for bi-linear forms based on an instance of
    [TorchOperator][pydvl.influence.torch.base.TensorOperator]. This means it
    computes weighted inner products of the form:

    $$ \langle \operatorname{Op}(x), y \rangle $$

    """

    def __init__(self, operator: TensorDictOperator):
        super().__init__(operator)

    def grads_inner_prod(
        self,
        left: TorchBatch,
        right: Optional[TorchBatch],
        gradient_provider: TorchGradientProvider,
    ) -> torch.Tensor:
        r"""
        Computes the gradient inner product of two batches of data, i.e.

        $$ \langle \nabla_{\omega}\ell(\omega, \text{left.x}, \text{left.y}),
        \nabla_{\omega}\ell(\omega, \text{right.x}, \text{right.y}) \rangle_{B}$$

        where $\nabla_{\omega}\ell(\omega, \cdot, \cdot)$ is represented by the
        `gradient_provider` and the expression must be understood sample-wise.

        Args:
            left: The first batch for gradient and inner product computation
            right: The second batch for gradient and inner product computation,
                optional; if not provided, the inner product will use the gradient
                computed for `left` for both arguments.
            gradient_provider: The gradient provider to compute the gradients.

        Returns:
            A tensor representing the inner products of the per-sample gradients
        """
        operator = cast(TensorDictOperator, self.operator)
        left_grads = gradient_provider.grads(left)
        if right is None:
            right_grads = left_grads
        else:
            right_grads = gradient_provider.grads(right)

        left_batch_size, right_batch_size = next(
            (
                (l.shape[0], r.shape[0])
                for r, l in zip(left_grads.values(), right_grads.values())
            )
        )

        if left_batch_size <= right_batch_size:
            left_grads = operator.apply_to_dict(left_grads)
            tensor_pairs = zip(left_grads.values(), right_grads.values())
        else:
            right_grads = operator.apply_to_dict(right_grads)
            tensor_pairs = zip(left_grads.values(), right_grads.values())

        tensors_to_reduce = (
            self._aggregate_grads(left, right) for left, right in tensor_pairs
        )

        return cast(torch.Tensor, sum(tensors_to_reduce))

    def mixed_grads_inner_prod(
        self,
        left: TorchBatch,
        right: Optional[TorchBatch],
        gradient_provider: TorchGradientProvider,
    ) -> torch.Tensor:
        r"""
        Computes the mixed gradient inner product of two batches of data, i.e.

        $$ \langle \nabla_{\omega}\ell(\omega, \text{left.x}, \text{left.y}),
        \nabla_{\omega}\nabla_{x}\ell(\omega, \text{right.x}, \text{right.y})
        \rangle_{B}$$

        where $\nabla_{\omega}\ell(\omega, \cdot)$ and
        $\nabla_{\omega}\nabla_{x}\ell(\omega, \cdot)$ are represented by the
        `gradient_provider`. The expression must be understood sample-wise.

        Args:
            left: The first batch for gradient and inner product computation
            right: The second batch for gradient and inner product computation
            gradient_provider: The gradient provider to compute the gradients.

        Returns:
            A tensor representing the inner products of the mixed per-sample gradients
        """
        operator = cast(TensorDictOperator, self.operator)
        if right is None:
            right = left
        right_grads = gradient_provider.mixed_grads(right)
        left_grads = gradient_provider.grads(left)
        left_grads = operator.apply_to_dict(left_grads)
        left_grads_views = (t.reshape(t.shape[0], -1) for t in left_grads.values())
        right_grads_views = (
            t.reshape(*right.x.shape, -1) for t in right_grads.values()
        )
        tensor_pairs = zip(left_grads_views, right_grads_views)
        tensors_to_reduce = (
            self._aggregate_mixed_grads(left, right) for left, right in tensor_pairs
        )
        return cast(torch.Tensor, sum(tensors_to_reduce))

    @staticmethod
    def _aggregate_mixed_grads(left: torch.Tensor, right: torch.Tensor):
        return torch.einsum("ik, j...k -> ij...", left, right)

    @staticmethod
    def _aggregate_grads(left: torch.Tensor, right: torch.Tensor):
        return torch.einsum("i..., j... -> ij", left, right)


class LowRankBilinearForm(OperatorBilinearForm):
    r"""
    Specialized bilinear form for operators of the type

    $$ \operatorname{Op}(b) = V D^{-1}V^Tb.$$

    It computes the expressions

    $$ \langle \operatorname{Op}(\nabla_{\theta} \ell(z, \theta)),
        \nabla_{\theta} \ell(z^{\prime}, \theta) \rangle =
        \langle V\nabla_{\theta} \ell(z, \theta),
        D^{-1}V\nabla_{\theta} \ell(z^{\prime}, \theta) \rangle$$

    in an efficient way using [torch.autograd][torch.autograd] functionality.
    """

    def __init__(self, operator: LowRankOperator):
        super().__init__(operator)

    def grads_inner_prod(
        self,
        left: TorchBatch,
        right: Optional[TorchBatch],
        gradient_provider: TorchGradientProvider,
    ) -> torch.Tensor:
        r"""
        Computes the gradient inner product of two batches of data, i.e.

        $$ \langle \nabla_{\omega}\ell(\omega, \text{left.x}, \text{left.y}),
        \nabla_{\omega}\ell(\omega, \text{right.x}, \text{right.y}) \rangle_{B}$$

        where $\nabla_{\omega}\ell(\omega, \cdot, \cdot)$ is represented by the
        `gradient_provider` and the expression must be understood sample-wise.

        Args:
            left: The first batch for gradient and inner product computation
            right: The second batch for gradient and inner product computation,
                optional; if not provided, the inner product will use the gradient
                computed for `left` for both arguments.
            gradient_provider: The gradient provider to compute the gradients.

        Returns:
            A tensor representing the inner products of the per-sample gradients
        """
        op = cast("LowRankOperator", self.operator)

        if op.exact:
            return super().grads_inner_prod(left, right, gradient_provider)

        V = op.low_rank_representation.projections
        D = op.low_rank_representation.eigen_vals.clone()
        regularization = op.regularization

        if regularization is not None:
            D += regularization

        V_left = gradient_provider.jacobian_prod(left, V.t())
        D_inv = 1.0 / D

        if right is None:
            V_right = V_left
        else:
            V_right = gradient_provider.jacobian_prod(right, V.t())

        V_right = V_right * D_inv.unsqueeze(-1)

        return torch.einsum("ij, ik -> jk", V_left, V_right)


OperatorBilinearFormType = TypeVar(
    "OperatorBilinearFormType", bound=OperatorBilinearForm
)


class TensorOperator(Operator[torch.Tensor, OperatorBilinearForm], ABC):
    """
    Abstract base class for operators that can be applied to instances of
    [torch.Tensor][torch.Tensor].
    """

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

    def _validate_tensor_input(self, tensor: torch.Tensor) -> None:
        if not (1 <= tensor.ndim <= 2):
            raise ValueError(
                f"Expected a 1 or 2 dimensional tensor, got {tensor.ndim} dimensions."
            )
        if tensor.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected the last dimension to be of size {self.input_size}."
            )

    def _apply(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return self._apply_to_mat(tensor.to(self.device))

        return self._apply_to_vec(tensor.to(self.device))

    @abstractmethod
    def _apply_to_vec(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Applies the operator to a single vector.
        Args:
            vec: A single vector consistent to the operator, i.e. it's length
                must be equal to the property `input_size`.

        Returns:
            A single vector after applying the batch operation
        """

    def _apply_to_mat(self, mat: torch.Tensor) -> torch.Tensor:
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
        return torch.func.vmap(self._apply_to_vec, in_dims=0, randomness="same")(mat)

    def as_bilinear_form(self) -> OperatorBilinearForm:
        return OperatorBilinearForm(self)


class TensorDictOperator(TensorOperator, ABC):
    """
    Abstract base class for operators that can be applied to instances of
    [torch.Tensor][torch.Tensor] and compatible dictionaries mapping strings to tensors.
    Input dictionaries must conform to the structure defined by the property
    `input_dict_structure`. Useful for operators involving autograd functionality
    to avoid intermediate flattening and concatenating of gradient inputs.
    """

    def apply_to_dict(self, mat: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Applies the operator to a dictionary of tensors, compatible to the structure
        defined by the property `input_dict_structure`.

        Args:
            mat: dictionary of tensors, whose keys and shapes match the property
                `input_dict_structure`.

        Returns:
            A dictionary of tensors after applying the operator
        """

        if not self._validate_mat_dict(mat):
            raise ValueError(
                f"Incompatible input structure, expected (excluding batch"
                f"dimension): \n {self.input_dict_structure}"
            )

        return self._apply_to_dict(self._dict_to_device(mat))

    def _dict_to_device(self, mat: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in mat.items()}

    @property
    @abstractmethod
    def input_dict_structure(self) -> Dict[str, Tuple[int, ...]]:
        """
        Implement this to expose the expected structure of the input tensor dict, i.e.
        a dictionary of shapes (excluding the first batch dimension), in order
        to validate the input tensor dicts.
        """

    @abstractmethod
    def _apply_to_dict(self, mat: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass

    def _validate_mat_dict(self, mat: Dict[str, torch.Tensor]) -> bool:
        for keys, val in mat.items():
            if val.shape[1:] != self.input_dict_structure[keys]:
                return False
        else:
            return True

    def as_bilinear_form(self) -> DictBilinearForm:
        return DictBilinearForm(self)


TorchOperatorType = TypeVar("TorchOperatorType", bound=TensorOperator)
"""Type variable bound to [TensorOperator][pydvl.influence.torch.base.TensorOperator]."""


class TorchOperatorGradientComposition(
    OperatorGradientComposition[
        torch.Tensor,
        TorchBatch,
        TorchOperatorType,
        TorchGradientProvider,
    ]
):
    """Represents a composable block that integrates a
    [TorchOperator][pydvl.influence.torch.base.TensorOperator] and
    a [TorchGradientProvider][pydvl.influence.torch.base.TorchGradientProvider]

    This block is designed to be flexible, handling different computational modes via
    an abstract operator and gradient provider.
    """

    def __init__(self, op: TorchOperatorType, gp: TorchGradientProvider):
        super().__init__(op, gp)

    def to(self, device: torch.device):
        self.gp = self.gp.to(device)
        self.op = self.op.to(device)
        return self

    def _tensor_inner_product(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("ia,j...a->ij...", left, right)


class TorchBlockMapper(
    BlockMapper[
        torch.Tensor, TorchBatch, TorchOperatorGradientComposition[TorchOperatorType]
    ]
):
    """
    Class for mapping operations across multiple compositional blocks represented by
    instances of [TorchOperatorGradientComposition]
    [pydvl.influence.torch.influence_function_model.TorchOperatorGradientComposition].

    This class takes a dictionary of compositional blocks and applies their methods to
    batches or tensors, and aggregates the results.
    """

    def __init__(
        self, composable_block_dict: OrderedDict[str, TorchOperatorGradientComposition]
    ):
        super().__init__(composable_block_dict)

    def _split_to_blocks(
        self, z: torch.Tensor, dim: int = -1
    ) -> OrderedDict[str, torch.Tensor]:
        block_sizes = [bi.op.input_size for bi in self.composable_block_dict.values()]

        block_dict = OrderedDict(
            zip(
                list(self.composable_block_dict.keys()),
                torch.split(z, block_sizes, dim=dim),
            )
        )
        return block_dict

    def to(self, device: torch.device):
        self.composable_block_dict = OrderedDict(
            [(k, bi.to(device)) for k, bi in self.composable_block_dict.items()]
        )
        return self


class TorchComposableInfluence(
    ComposableInfluence[
        torch.Tensor, TorchBatch, DataLoader, TorchBlockMapper[TorchOperatorType]
    ],
    ModelInfoMixin,
    ABC,
):
    """
    Abstract base class, that allow for block-wise computation of influence
    quantities with the [torch][torch] framework.
    Inherit from this base class for specific influence algorithms.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        block_structure: Union[BlockMode, OrderedDict[str, List[str]]] = BlockMode.FULL,
        regularization: Optional[Union[float, Dict[str, Optional[float]]]] = None,
    ):
        parameter_dict_builder = ModelParameterDictBuilder(model)
        if isinstance(block_structure, BlockMode):
            self.parameter_dict = parameter_dict_builder.build_from_block_mode(
                block_structure
            )
        else:
            self.parameter_dict = parameter_dict_builder.build(block_structure)

        self._regularization_dict = self._build_regularization_dict(regularization)

        super().__init__(model)

    def _concat(self, tensors: Iterable[torch.Tensor], dim: int):
        return torch.cat(list(tensors), dim=dim)

    def _flatten_trailing_dim(self, tensor: torch.Tensor):
        return tensor.reshape((tensor.shape[0], -1))

    @property
    def block_names(self) -> List[str]:
        return list(self.parameter_dict.keys())

    @property
    def n_parameters(self):
        return sum(
            param.numel()
            for block in self.parameter_dict.values()
            for param in block.values()
        )

    @abstractmethod
    def with_regularization(
        self, regularization: Union[float, Dict[str, Optional[float]]]
    ) -> TorchComposableInfluence:
        pass

    def _build_regularization_dict(
        self, regularization: Optional[Union[float, Dict[str, Optional[float]]]]
    ) -> Dict[str, Optional[float]]:
        if regularization is None or isinstance(regularization, float):
            return {
                k: self._validate_regularization(k, regularization)
                for k in self.block_names
            }

        if set(regularization.keys()).issubset(set(self.block_names)):
            raise ValueError(
                f"The regularization must be a float or the keys of the regularization"
                f"dictionary must match a subset of"
                f"block names: \n {self.block_names}.\n Found not in block names: \n"
                f"{set(regularization.keys()).difference(set(self.block_names))}"
            )
        return {
            k: self._validate_regularization(k, regularization.get(k, None))
            for k in self.block_names
        }

    @staticmethod
    def _validate_regularization(
        block_name: str, value: Optional[float]
    ) -> Optional[float]:
        if isinstance(value, float) and value < 0.0:
            raise ValueError(
                f"The regularization for block '{block_name}' must be non-negative, "
                f"but found {value=}"
            )
        return value

    @abstractmethod
    def _create_block(
        self,
        block_params: Dict[str, torch.nn.Parameter],
        data: DataLoader,
        regularization: Optional[float],
    ) -> TorchOperatorGradientComposition:
        pass

    def _create_block_mapper(self, data: DataLoader) -> TorchBlockMapper:
        block_influence_dict = OrderedDict()
        for k, p in self.parameter_dict.items():
            reg = self._regularization_dict.get(k, None)
            reg = self._validate_regularization(k, reg)
            block_influence_dict[k] = self._create_block(p, data, reg).to(self.device)

        return TorchBlockMapper(block_influence_dict)

    @staticmethod
    def _create_batch(x: torch.Tensor, y: torch.Tensor) -> TorchBatch:
        return TorchBatch(x, y)

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        if hasattr(self, "block_mapper") and self.block_mapper is not None:
            self.block_mapper = self.block_mapper.to(device)
        return self

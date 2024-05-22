from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Dict, Callable, Union, List

import torch
from torch.func import functional_call
from torch.utils.data import DataLoader

from .functional import create_per_sample_gradient_function, \
    create_per_sample_mixed_derivative_function, create_matrix_jacobian_product_function
from .util import LossType, ModelParameterDictBuilder, \
    BlockMode, flatten_dimensions, ModelInfoMixin
from ..base_influence_function_model import ComposableInfluence
from ..types import PerSampleGradientProvider, Operator, BilinearForm, Batch, \
    OperatorGradientComposition, BlockMapper


@dataclass(frozen=True)
class TorchBatch(Batch):
    """
    A convenience class for handling batches of data. Validates, the alignment
    of the first dimension (batch dimension) of the input and target tensor

    Attributes:
        x: The input tensor that contains features or data points.
        y: The target tensor that contains labels corresponding to the inputs.

    """

    x: torch.Tensor
    y: torch.Tensor

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


class TorchPerSampleGradientProvider(
    PerSampleGradientProvider[TorchBatch, torch.Tensor], ABC
):
    r"""
    Abstract base class for calculating per-sample gradients of a function defined by
    a [torch.nn.Module][torch.nn.Module] and a loss function.

    This class must be subclassed with implementations for its abstract methods tailored
    to specific gradient computation needs, e.g. using [torch.autograd][torch.autograd]
    or stochastic finite differences.

    Consider a function

    $$ \ell: \mathbb{R}^{d_1} \times \mathbb{R}^{d_2} \times \mathbb{R}^{n} \times
        \mathbb{R}^{n}, \quad \ell(\omega_1, \omega_2, x, y) =
        \operatorname{loss}(f(\omega_1, \omega_2; x), y) $$

    e.g. a two layer neural network $f$ with a loss function, then this object should
    compute the expressions:

    $$ \nabla_{\omega_{i}}\ell(\omega_1, \omega_2, x, y),
    \nabla_{\omega_{i}}\nabla_{x}\ell(\omega_1, \omega_2, x, y),
    \nabla_{\omega}\ell(\omega_1, \omega_2, x, y) \cdot v$$

    """

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
        r"""
        Computes and returns a dictionary mapping gradient names to their respective
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
        gradient_dict = self._per_sample_gradient_dict(batch.to(self.device))
        return self._detach_dict(gradient_dict)

    def per_sample_mixed_gradient_dict(
        self, batch: TorchBatch
    ) -> Dict[str, torch.Tensor]:
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
        gradient_dict = self._per_sample_mixed_gradient_dict(batch.to(self.device))
        return self._detach_dict(gradient_dict)

    def matrix_jacobian_product(
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
    r"""
    Compute per-sample gradients of a function defined by
    a [torch.nn.Module][torch.nn.Module] and a loss function using
    [torch.func][torch.func].

    Consider a function

    $$ \ell: \mathbb{R}^{d_1} \times \mathbb{R}^{d_2} \times \mathbb{R}^{n} \times
        \mathbb{R}^{n}, \quad \ell(\omega_1, \omega_2, x, y) =
        \operatorname{loss}(f(\omega_1, \omega_2; x), y) $$

    e.g. a two layer neural network $f$ with a loss function, then this object should
    compute the expressions:

    $$ \nabla_{\omega_{i}}\ell(\omega_1, \omega_2, x, y),
    \nabla_{\omega_{i}}\nabla_{x}\ell(\omega_1, \omega_2, x, y),
    \nabla_{\omega}\ell(\omega_1, \omega_2, x, y) \cdot v$$

    """

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


class OperatorBilinearForm(
    BilinearForm[torch.Tensor, TorchBatch, TorchPerSampleGradientProvider]
):
    r"""
    Base class for bilinear forms based on an instance of
    [TorchOperator][pydvl.influence.torch.operator.base.TorchOperator]. This means it
    computes weighted inner products of the form:

    $$ \langle \operatorname{Op}(x), y \rangle $$

    """

    def __init__(
        self,
        operator: "TorchOperator",
    ):
        self.operator = operator

    def inner_product(
        self, left: torch.Tensor, right: Optional[torch.Tensor]
    ) -> torch.Tensor:
        r"""
        Computes the weighted inner product of two vectors, i.e.

        $$ \langle x, y \rangle_{B} = \langle \operatorname{Op}(x), y \rangle $$

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
        left_result = self.operator.apply_to_mat(left)

        if left_result.ndim == right.ndim and left.shape[-1] == right.shape[-1]:
            return left_result @ right.T

        return torch.einsum("ia,j...a->ij...", left_result, right)


class TorchOperator(Operator[torch.Tensor, OperatorBilinearForm], ABC):
    def __init__(self, regularization: float = 0.0):
        """
        Initializes the Operator with an optional regularization parameter.

        Args:
            regularization: A non-negative float that represents the regularization
                strength (default is 0.0).

        Raises:
            ValueError: If the regularization parameter is negative.
        """
        if regularization < 0:
            raise ValueError("regularization must be non-negative")
        self._regularization = regularization

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
    def _apply_to_vec(self, vec: torch.Tensor) -> torch.Tensor:
        pass

    def as_bilinear_form(self):
        return OperatorBilinearForm(self)

    def apply_to_vec(self, vec: torch.Tensor) -> torch.Tensor:
        return self._apply_to_vec(vec.to(self.device))

    def apply_to_mat(self, mat: torch.Tensor) -> torch.Tensor:
        return torch.func.vmap(self.apply_to_vec, in_dims=0, randomness="same")(mat)


class TorchOperatorGradientComposition(
    OperatorGradientComposition[
        torch.Tensor, TorchBatch, TorchOperator, TorchPerSampleGradientProvider
    ]
):
    """
    Representing a composable block that integrates an [TorchOperator]
    [pydvl.influence.torch.operator.base.TorchOperator] and
    a [TorchPerSampleGradientProvider]
    [pydvl.influence.torch.operator.gradient_provider.TorchPerSampleGradientProvider]

    This block is designed to be flexible, handling different computational modes via
    an abstract operator and gradient provider.
    """

    def __init__(self, op: TorchOperator, gp: TorchPerSampleGradientProvider):
        super().__init__(op, gp)

    def to(self, device: torch.device):
        self.gp = self.gp.to(device)
        self.op = self.op.to(device)
        return self


class TorchBlockMapper(
    BlockMapper[torch.Tensor, TorchBatch, TorchOperatorGradientComposition]
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
    ComposableInfluence[torch.Tensor, TorchBatch, DataLoader, TorchBlockMapper],
    ModelInfoMixin,
    ABC,
):
    def __init__(
        self,
        model: torch.nn.Module,
        block_structure: Union[
            BlockMode, OrderedDict[str, OrderedDict[str, torch.nn.Parameter]]
        ] = BlockMode.FULL,
        regularization: Optional[Union[float, Dict[str, Optional[float]]]] = None,
    ):
        if isinstance(block_structure, BlockMode):
            self.parameter_dict = ModelParameterDictBuilder(model).build(
                block_structure
            )
        else:
            self.parameter_dict = block_structure

        self._regularization_dict = self._build_regularization_dict(regularization)

        super().__init__(model)

    @property
    def block_names(self) -> List[str]:
        return list(self.parameter_dict.keys())

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

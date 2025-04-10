"""
This module offers a set of generic types, which can be used to build modular and
flexible components for influence computation for different tensor frameworks.


Key components include:

1. [GradientProvider][pydvl.influence.types.GradientProvider]: A generic
    abstract base class designed to provide methods for computing per-sample
    gradients and other related computations for given data batches.

2. [BilinearForm][pydvl.influence.types.BilinearForm]: A generic abstract base class
    for representing bilinear forms for computing inner products involving gradients.

3. [Operator][pydvl.influence.types.Operator]: A generic abstract base class for
    operators that can apply transformations to vectors and matrices and can be
    represented as bilinear forms.

4. [OperatorGradientComposition][pydvl.influence.types.OperatorGradientComposition]: A
    generic abstract composition class that integrates an operator with a gradient
    provider to compute interactions between batches of data.

5. [BlockMapper][pydvl.influence.types.BlockMapper]: A generic abstract base class
    for mapping operations across multiple compositional blocks, given by objects
    of type
    [OperatorGradientComposition][pydvl.influence.types.OperatorGradientComposition],
    and aggregating the results.

To see the usage of these types, see the implementation
[ComposableInfluence][pydvl.influence.base_influence_function_model.ComposableInfluence]
. Using these components allows the straightforward implementation of various
combinations of approximations of inverse Hessian applications
(or Gauss-Newton approximations), different blocking strategies
(e.g. layer-wise or block-wise) and different ways to
compute gradients.

For the usage with a specific tensor framework, these types must be subclassed. An
example for [torch][torch] is provided in the module
[pydvl.influence.torch.base][pydvl.influence.torch.base] and the base class
[TorchComposableInfluence][pydvl.influence.torch.base.TorchComposableInfluence].
"""

from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Collection,
    Generator,
    Generic,
    Iterable,
    Optional,
    OrderedDict,
    TypeVar,
    Union,
    cast,
)


class InfluenceMode(str, Enum):
    """
    Enum representation for the types of influence.

    Attributes:
        Up: [Approximating the influence of a point]
            [influence-of-a-point]
        Perturbation: [Perturbation definition of the influence score]
            [perturbation-definition-of-the-influence-score]

    """

    Up = "up"
    Perturbation = "perturbation"


"""Type variable for tensors, i.e. sequences of numbers"""
TensorType = TypeVar("TensorType", bound=Collection)
DataLoaderType = TypeVar("DataLoaderType", bound=Iterable)


@dataclass(frozen=True)
class Batch(Generic[TensorType]):
    """
    Represents a batch of data containing features and labels.

    Attributes:
        x: Represents the input features of the batch.
        y: Represents the labels or targets associated with the input features.
    """

    x: TensorType
    y: TensorType


BatchType = TypeVar("BatchType", bound=Batch)


class GradientProvider(Generic[BatchType, TensorType], ABC):
    r"""
    Provides an interface for calculating per-sample gradients and other related
    computations for a given batch of data.

    This class must be subclassed with implementations for its abstract methods tailored
    to specific gradient computation needs, e.g. using an autograd engine for
    a model loss function. Consider a function

    $$ \ell: \mathbb{R}^{d_1} \times \mathbb{R}^{d_2} \times \mathbb{R}^{n} \times
        \mathbb{R}^{n}, \quad \ell(\omega_1, \omega_2, x, y) =
        \operatorname{loss}(f(\omega_1, \omega_2; x), y) $$

    e.g. a two layer neural network $f$ with a loss function, then this object should
    compute the expressions:

    $$ \nabla_{\omega_{i}}\ell(\omega_1, \omega_2, x, y),
    \nabla_{\omega_{i}}\nabla_{x}\ell(\omega_1, \omega_2, x, y),
    \nabla_{\omega}\ell(\omega_1, \omega_2, x, y) \cdot v$$

    """

    @abstractmethod
    def jacobian_prod(
        self,
        batch: BatchType,
        g: TensorType,
    ) -> TensorType:
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

    @abstractmethod
    def flat_grads(self, batch: BatchType) -> TensorType:
        r"""
        Computes and returns the flat per-sample gradients for the provided batch.
        Given the example in the class docstring, this means

        $$ (\nabla_{\omega_{1}}\ell(\omega_1, \omega_2,
            \text{batch.x}, \text{batch.y}),
            \nabla_{\omega_{2}}\ell(\omega_1, \omega_2,
            \text{batch.x}, \text{batch.y}))$$

        where the first dimension of the resulting tensor is always considered to be
        the batch dimension, so the shape of the resulting tensor is $(N, d_1+d_2)$,
        where $N$ is the number of samples in the batch.

        Args:
            batch: The batch of data for which to compute the gradients.

        Returns:
            A tensor containing the flat gradients computed per sample.
        """

    @abstractmethod
    def flat_mixed_grads(self, batch: BatchType) -> TensorType:
        r"""
        Computes and returns the flat per-sample mixed gradients for the provided batch.
        Given the example in the class docstring, this means

        $$ (\nabla_{\omega_1}\nabla_{x}\ell(\omega_1,
            \omega_2, \text{batch.x}, \text{batch.y}),
            \nabla_{\omega_1}\nabla_{x}\ell(\omega_1,
            \omega_2, \text{batch.x}, \text{batch.y} ))$$

        where the first dimension of the resulting tensor is always considered to be
        the batch dimension and the last to be the non-batch input related derivatives.
        So the shape of the resulting tensor is $(N, n, d_1 + d_2)$,
        where $N$ is the number of samples in the batch.

        Args:
            batch: The batch of data for which to compute the flat mixed gradients.

        Returns:
            A tensor containing the flat mixed gradients computed per sample.
        """


GradientProviderType = TypeVar("GradientProviderType", bound=GradientProvider)


class BilinearForm(Generic[TensorType, BatchType, GradientProviderType], ABC):
    """
    Abstract base class for bilinear forms, which facilitates the computation of inner
    products involving gradients of batches of data.
    """

    @abstractmethod
    def inner_prod(self, left: TensorType, right: Optional[TensorType]) -> TensorType:
        r"""
        Computes the inner product of two vectors, i.e.

        $$ \langle x, y \rangle_{B}$$

        if we denote the bilinear-form by $\langle \cdot, \cdot \rangle_{B}$.
        The implementations must take care of according vectorization to make
        it applicable to the case, where `left` and `right` are not one-dimensional.
        In this case, the trailing dimension of the `left` and `right` tensors are
        considered for the computation of the inner product. For example,
        if `left` is a tensor of shape $(N, D)$ and, `right` is of shape $(M,..., D)$,
        then the result is of shape $(N, M, ...)$.

        Args:
            left: The first tensor in the inner product computation.
            right: The second tensor, optional; if not provided, the inner product will
                use `left` tensor for both arguments.

        Returns:
            A tensor representing the inner product.
        """

    def grads_inner_prod(
        self,
        left: BatchType,
        right: Optional[BatchType],
        gradient_provider: GradientProviderType,
    ) -> TensorType:
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
        left_grad = gradient_provider.flat_grads(left)
        if right is None:
            right_grad = left_grad
        else:
            right_grad = gradient_provider.flat_grads(right)
        return self.inner_prod(left_grad, right_grad)

    def mixed_grads_inner_prod(
        self,
        left: BatchType,
        right: Optional[BatchType],
        gradient_provider: GradientProviderType,
    ) -> TensorType:
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
        left_grad = gradient_provider.flat_grads(left)
        if right is None:
            right = left
        right_mixed_grad = gradient_provider.flat_mixed_grads(right)
        return self.inner_prod(left_grad, right_mixed_grad)


BilinearFormType = TypeVar("BilinearFormType", bound=BilinearForm)


class Operator(Generic[TensorType, BilinearFormType], ABC):
    """
    Abstract base class for operators, capable of applying transformations to
    vectors and matrices, and can be represented as a bilinear form.
    """

    @property
    @abstractmethod
    def input_size(self) -> int:
        """
        Abstract property to get the needed size for inputs to the operator
        instance

        Returns:
            An integer representing the input size.
        """

    @abstractmethod
    def _validate_tensor_input(self, tensor: TensorType) -> None:
        """
        Validates the input tensor for the operator.

        Args:
            tensor: A tensor to validate.

        Raises:
            ValueError: If the tensor is invalid for the operator.
        """

    def apply(self, tensor: TensorType) -> TensorType:
        """
        Applies the operator to a tensor.

        Args:
            tensor: A tensor, whose tailing dimension must conform to the
                operator's input size

        Returns:
            A tensor representing the result of the operator application.
        """
        self._validate_tensor_input(tensor)
        return self._apply(tensor)

    @abstractmethod
    def _apply(self, tensor: TensorType) -> TensorType:
        """
        Applies the operator to a tensor. Implement this to handle
        batched input.

        Args:
            tensor: A tensor, whose tailing dimension must conform to the
                operator's input size

        Returns:
            A tensor representing the result of the operator application.
        """

    @abstractmethod
    def as_bilinear_form(self) -> BilinearFormType:
        r"""
        Represents the operator as a bilinear form, i.e. the weighted inner product

        $$ \langle \operatorname{Op}(x), y \rangle$$

        Returns:
            An instance of type [BilinearForm][pydvl.influence.types.BilinearForm]
                representing this operator.
        """


OperatorType = TypeVar("OperatorType", bound=Operator)


class OperatorGradientComposition(
    Generic[
        TensorType,
        BatchType,
        OperatorType,
        GradientProviderType,
    ]
):
    """
    Generic base class representing a composable block that integrates an operator and
    a gradient provider to compute interactions between batches of data.

    This block is designed to be flexible, handling different computational modes via
    an abstract operator and gradient provider.

    Attributes:
        op: The operator used for transformations and influence computations.
        gp: The gradient provider used for obtaining necessary gradients.
    """

    def __init__(self, op: OperatorType, gp: GradientProviderType):
        self.gp = gp
        self.op = op

    @abstractmethod
    def _tensor_inner_product(self, left: TensorType, right: TensorType) -> TensorType:
        """Implement this method in a way such that the aggregation of the tensors
        is represented by the Einstein summation convention ia,j...a -> ij..."""

    def interactions(
        self,
        left_batch: BatchType,
        right_batch: Optional[BatchType],
        mode: InfluenceMode,
    ) -> TensorType:
        r"""
        Computes the interaction between the gradients on two batches of data based on
        the specified mode weighted by the operator action,
        i.e.

        $$ \langle \operatorname{Op}(\nabla_{\omega}\ell(\omega, \text{left.x},
        \text{left.y})),
        \nabla_{\omega}\ell(\omega, \text{right.x}, \text{right.y}) \rangle$$

        for the case `InfluenceMode.Up` and

        $$ \langle \operatorname{Op}(\nabla_{\omega}\ell(\omega, \text{left.x},
        \text{left.y})),
        \nabla_{\omega}\nabla_{x}\ell(\omega, \text{right.x}, \text{right.y}) \rangle $$

        for the case `InfluenceMode.Perturbation`.

        Args:
            left_batch: The left data batch for gradient computation.
            right_batch: The right data batch for gradient computation.
            mode: An instance of InfluenceMode determining the type of influence
                computation.

        Returns:
            The result of the influence computation as dictated by the mode.
        """
        bilinear_form = self.op.as_bilinear_form()
        if mode == InfluenceMode.Up:
            return cast(
                TensorType,
                bilinear_form.grads_inner_prod(left_batch, right_batch, self.gp),
            )
        elif mode == InfluenceMode.Perturbation:
            return cast(
                TensorType,
                bilinear_form.mixed_grads_inner_prod(left_batch, right_batch, self.gp),
            )
        else:
            raise UnsupportedInfluenceModeException(mode)

    def transformed_grads(self, batch: BatchType) -> TensorType:
        r"""
        Computes the gradients of a data batch, transformed by the operator application
        , i.e. the expressions

        $$ \operatorname{Op}(\nabla_{\omega}\ell(\omega, \text{batch.x},
            \text{batch.y})) $$

        Args:
            batch: The data batch for gradient computation.

        Returns:
            A tensor representing the application of the operator to the gradients.

        """
        grads = self.gp.flat_grads(batch)
        return cast(TensorType, self.op.apply(grads))

    def interactions_from_transformed_grads(
        self, left_factors: TensorType, right_batch: BatchType, mode: InfluenceMode
    ) -> TensorType:
        r"""
        Computes the interaction between the transformed gradients on two batches of
        data using pre-computed factors and a batch of data,
        based on the specified mode. This means

        $$ \langle \text{left_factors},
        \nabla_{\omega}\ell(\omega, \text{right.x}, \text{right.y}) \rangle$$

        for the case `InfluenceMode.Up` and

        $$ \langle \text{left_factors},
        \nabla_{\omega}\nabla_{x}\ell(\omega, \text{right.x}, \text{right.y}) \rangle $$

        for the case `InfluenceMode.Perturbation`.

        Args:
            left_factors: Pre-computed tensor factors from a left batch.
            right_batch: The right data batch for influence computation.
            mode: An instance of InfluenceMode determining the type of influence
                computation.

        Returns:
            The result of the interaction computation using the provided factors and
                batch gradients.
        """
        if mode is InfluenceMode.Up:
            right_grads = self.gp.flat_grads(right_batch)
        else:
            right_grads = self.gp.flat_mixed_grads(right_batch)
        return self._tensor_inner_product(left_factors, right_grads)


OperatorGradientCompositionType = TypeVar(
    "OperatorGradientCompositionType", bound=OperatorGradientComposition
)


class BlockMapper(Generic[TensorType, BatchType, OperatorGradientCompositionType], ABC):
    """
    Abstract base class for mapping operations across multiple compositional blocks.

    This class takes a dictionary of compositional blocks and applies their methods to
    batches or tensors, and aggregates the results.

    Attributes:
        composable_block_dict: A dictionary mapping string identifiers to
            composable blocks which define operations like transformations and
            interactions.
    """

    def __init__(
        self, composable_block_dict: OrderedDict[str, OperatorGradientCompositionType]
    ):
        self.composable_block_dict = composable_block_dict

    def __getitem__(self, item: str):
        return self.composable_block_dict[item]

    def items(self):
        return self.composable_block_dict.items()

    def _to_ordered_dict(
        self, tensor_generator: Generator[TensorType, None, None]
    ) -> OrderedDict[str, TensorType]:
        tensor_dict = collections.OrderedDict()
        for k, t in zip(self.composable_block_dict.keys(), tensor_generator):
            tensor_dict[k] = t
        return tensor_dict

    @abstractmethod
    def _split_to_blocks(
        self, z: TensorType, dim: int = -1
    ) -> OrderedDict[str, TensorType]:
        """Must be implemented in a way to preserve the ordering defined by the
        `composable_block_dict` attribute"""

    def transformed_grads(
        self,
        batch: BatchType,
    ) -> OrderedDict[str, TensorType]:
        """
        Computes and returns the transformed gradients for a batch in dictionary
        with the keys defined by the block names.

        Args:
            batch: The batch of data for which to compute transformed gradients.

        Returns:
            An ordered dictionary of transformed gradients by block.
        """
        tensor_gen = self.generate_transformed_grads(batch)
        return self._to_ordered_dict(tensor_gen)

    def interactions(
        self, left_batch: BatchType, right_batch: BatchType, mode: InfluenceMode
    ) -> OrderedDict[str, TensorType]:
        """
        Computes interactions between two batches, aggregated by block,
        based on a specified mode.

        Args:
            left_batch: The left batch for interaction computation.
            right_batch: The right batch for interaction computation.
            mode: The mode determining the type of interactions.

        Returns:
            An ordered dictionary of gradient interactions by block.
        """
        tensor_gen = self.generate_interactions(left_batch, right_batch, mode)
        return self._to_ordered_dict(tensor_gen)

    def interactions_from_transformed_grads(
        self,
        left_factors: OrderedDict[str, TensorType],
        right_batch: BatchType,
        mode: InfluenceMode,
    ) -> OrderedDict[str, TensorType]:
        """
        Computes interactions from transformed gradients and a right batch,
        aggregated by block and based on a mode.

        Args:
            left_factors: Pre-computed factors as a tensor or an ordered dictionary of
                tensors by block. If the input is a tensor, it is split into blocks
                according to the ordering in the `composable_block_dict` attribute.
            right_batch: The right batch for interaction computation.
            mode: The mode determining the type of interactions.

        Returns:
            An ordered dictionary of interactions from transformed gradients by block.
        """
        tensor_gen = self.generate_interactions_from_transformed_grads(
            left_factors, right_batch, mode
        )
        return self._to_ordered_dict(tensor_gen)

    def generate_transformed_grads(
        self, batch: BatchType
    ) -> Generator[TensorType, None, None]:
        """
        Generator that yields transformed gradients for a given batch,
        processed by each block.

        Args:
            batch: The batch of data for which to generate transformed gradients.

        Yields:
            Transformed gradients for each block.
        """
        for comp_block in self.composable_block_dict.values():
            yield comp_block.transformed_grads(batch)

    def generate_interactions(
        self,
        left_batch: BatchType,
        right_batch: Optional[BatchType],
        mode: InfluenceMode,
    ) -> Generator[TensorType, None, None]:
        """
        Generator that yields gradient interactions between two batches, processed by
        each block based on a mode.

        Args:
            left_batch: The left batch for interaction computation.
            right_batch: The right batch for interaction computation.
            mode: The mode determining the type of interactions.

        Yields:
            TensorType: Gradient interactions for each block.
        """
        for comp_block in self.composable_block_dict.values():
            yield comp_block.interactions(left_batch, right_batch, mode)

    def generate_interactions_from_transformed_grads(
        self,
        left_factors: Union[TensorType, OrderedDict[str, TensorType]],
        right_batch: BatchType,
        mode: InfluenceMode,
    ) -> Generator[TensorType, None, None]:
        """
        Generator that yields interactions computed from pre-computed factors and a
        right batch, processed by each block based on a mode.

        Args:
            left_factors: Pre-computed factors as a tensor or an ordered dictionary of
                tensors by block.
            right_batch: The right batch for interaction computation.
            mode: The mode determining the type of interactions.

        Yields:
            TensorType: Interactions for each block.
        """
        if not isinstance(left_factors, dict):
            left_factors_dict = self._split_to_blocks(left_factors)
        else:
            left_factors_dict = cast(OrderedDict[str, TensorType], left_factors)
        for k, comp_block in self.composable_block_dict.items():
            yield comp_block.interactions_from_transformed_grads(
                left_factors_dict[k], right_batch, mode
            )


BlockMapperType = TypeVar("BlockMapperType", bound=BlockMapper)


class UnsupportedInfluenceModeException(ValueError):
    def __init__(self, mode: str):
        super().__init__(
            f"Provided {mode=} is not supported. Choose one of InfluenceMode.Up "
            f"and InfluenceMode.Perturbation"
        )

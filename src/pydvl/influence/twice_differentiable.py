from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Collection,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

__all__ = [
    "DataLoaderType",
    "ModelType",
    "TensorType",
    "InverseHvpResult",
    "TwiceDifferentiable",
    "TensorUtilities",
]

TensorType = TypeVar("TensorType", bound=Collection)
"""Type variable for tensors, i.e. sequences of numbers"""

ModelType = TypeVar("ModelType", bound="TwiceDifferentiable")
"""Type variable for twice differentiable models"""

DataLoaderType = TypeVar("DataLoaderType", bound=Iterable)
"""Type variable for data loaders"""


@dataclass(frozen=True)
class InverseHvpResult(Generic[TensorType]):
    r"""
    Container class for results of solving a problem \(Ax=b\)

    Args:
        x: solution of a problem \(Ax=b\)
        info: additional information, to couple with the solution itself
    """
    x: TensorType
    info: Dict[str, Any]

    def __iter__(self):
        return iter((self.x, self.info))


class TwiceDifferentiable(ABC, Generic[TensorType]):
    """
    Abstract base class for wrappers of differentiable models and losses. Meant to be subclassed for each
    supported framework.
    Provides methods to compute gradients and second derivative of the loss wrt. the model parameters
    """

    @classmethod
    @abstractmethod
    def tensor_type(cls):
        pass

    @property
    @abstractmethod
    def num_params(self) -> int:
        """Returns the number of parameters of the model"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> List[TensorType]:
        """Returns all the model parameters that require differentiation"""
        pass

    @abstractmethod
    def grad(
        self, x: TensorType, y: TensorType, create_graph: bool = False
    ) -> TensorType:
        r"""
        Calculates gradient of model parameters with respect to the model parameters.

        Args:
            x: A matrix representing the features \(x_i\).
            y: A matrix representing the target values \(y_i\).
            create_graph: Used for further differentiation on input parameters.

        Returns:
            An array with the gradients of the model.
        """

        pass

    @abstractmethod
    def hessian(self, x: TensorType, y: TensorType) -> TensorType:
        r"""
        Calculates the full Hessian of \(L(f(x),y)\) with respect to the model parameters given data \(x\) and \(y\).

        Args:
            x: An array representing the features \(x_i\).
            y: An array representing the target values \(y_i\).

        Returns:
            A tensor representing the Hessian of the model, i.e. the second derivative
                with respect to the model parameters.
        """

        pass

    @staticmethod
    @abstractmethod
    def mvp(
        grad_xy: TensorType,
        v: TensorType,
        backprop_on: Union[TensorType, Sequence[TensorType]],
        *,
        progress: bool = False,
    ) -> TensorType:
        r"""
        Calculates the second order derivative of the model along directions \(v\).
        The second order derivative can be selected through the `backprop_on` argument.

        Args:
            grad_xy: An array [P] holding the gradients of the model parameters with respect to input \(x\) and
                labels \(y\). \(P\) is the number of parameters of the model. Typically obtained through `self.grad`.
            v: An array ([DxP] or even one-dimensional [D]) which multiplies the matrix.
                \(D\) is the number of directions.
            progress: If `True`, progress is displayed.
            backprop_on: Tensor used in the second backpropagation. The first one is along \(x\) and \(y\)
                as defined via `grad_xy`.

        Returns:
            A matrix representing the implicit matrix-vector product of the model along the given directions.
                Output shape is [DxM], where \(M\) is the number of elements of `backprop_on`.
        """
        pass


class TensorUtilities(Generic[TensorType, ModelType, DataLoaderType], ABC):
    twice_differentiable_type: Type[TwiceDifferentiable]
    registry: Dict[Type[TwiceDifferentiable], Type["TensorUtilities"]] = {}

    def __init_subclass__(cls, **kwargs):
        """
        Automatically registers non-abstract subclasses in the registry.

        This method checks if `twice_differentiable_type` is defined in the subclass and if it is of the correct type.
        If either attribute is missing or incorrect, a `TypeError` is raised.

        Args:
            kwargs: Additional keyword arguments.

        Raises:
            TypeError: If the subclass does not define `twice_differentiable_type`, or if it is not of the correct type.
        """

        if not hasattr(cls, "twice_differentiable_type") or not isinstance(
            cls.twice_differentiable_type, type
        ):
            raise TypeError(
                f"'twice_differentiable_type' must be a Type[TwiceDifferentiable]"
            )

        cls.registry[cls.twice_differentiable_type] = cls

        super().__init_subclass__(**kwargs)

    @staticmethod
    @abstractmethod
    def data_loader_to_tensor_tuple(
        data_loader: DataLoaderType,
    ) -> Tuple[TensorType, TensorType]:
        pass

    @staticmethod
    @abstractmethod
    def einsum(equation, *operands) -> TensorType:
        """Sums the product of the elements of the input `operands` along dimensions specified using a notation
        based on the Einstein summation convention.
        """

    @staticmethod
    @abstractmethod
    def cat(a: Sequence[TensorType], **kwargs) -> TensorType:
        """Concatenates a sequence of tensors into a single torch tensor"""

    @staticmethod
    @abstractmethod
    def stack(a: Sequence[TensorType], **kwargs) -> TensorType:
        """Stacks a sequence of tensors into a single torch tensor"""

    @staticmethod
    @abstractmethod
    def unsqueeze(x: TensorType, dim: int) -> TensorType:
        """Add a singleton dimension at a specified position in a tensor"""

    @staticmethod
    @abstractmethod
    def get_element(x: TensorType, idx: int) -> TensorType:
        """Get the tensor element x[i] from the first non-singular dimension"""

    @staticmethod
    @abstractmethod
    def slice(x: TensorType, start: int, stop: int, axis: int = 0) -> TensorType:
        """Slice a tensor in the provided axis"""

    @staticmethod
    @abstractmethod
    def shape(x: TensorType) -> Tuple[int, ...]:
        """Slice a tensor in the provided axis"""

    @staticmethod
    @abstractmethod
    def reshape(x: TensorType, shape: Tuple[int, ...]) -> TensorType:
        """Reshape a tensor to the provided shape"""

    @staticmethod
    @abstractmethod
    def cat_gen(
        a: Generator[TensorType, None, None],
        resulting_shape: Tuple[int, ...],
        model: ModelType,
    ) -> TensorType:
        """Concatenate tensors from a generator. Resulting tensor is of shape resulting_shape
        and compatible to model
        """

    @classmethod
    def from_twice_differentiable(
        cls,
        twice_diff: TwiceDifferentiable,
    ) -> Type["TensorUtilities"]:
        """
        Factory method to create an instance of a subclass
        [TensorUtilities][pydvl.influence.twice_differentiable.TensorUtilities] from an instance of a subclass of
        [TwiceDifferentiable][pydvl.influence.twice_differentiable.TwiceDifferentiable].

        Args:
            twice_diff: An instance of a subclass of
                [TwiceDifferentiable][pydvl.influence.twice_differentiable.TwiceDifferentiable]
                for which a corresponding [TensorUtilities][pydvl.influence.twice_differentiable.TensorUtilities]
                object is required.

        Returns:
            An subclass of [TensorUtilities][pydvl.influence.twice_differentiable.TensorUtilities]
                registered to the provided subclass instance of
                [TwiceDifferentiable][pydvl.influence.twice_differentiable.TwiceDifferentiable] object.

        Raises:
            KeyError: If there's no registered [TensorUtilities][pydvl.influence.twice_differentiable.TensorUtilities]
                for the provided [TwiceDifferentiable][pydvl.influence.twice_differentiable.TwiceDifferentiable] type.
        """

        tu = cls.registry.get(type(twice_diff), None)

        if tu is None:
            raise KeyError(
                f"No registered TensorUtilities for the type {type(twice_diff).__name__}"
            )

        return tu


class InfluenceType(str, Enum):
    r"""
    Enum representation for the types of influence.

    Attributes:
        Up: Up-weighting a training point, see section 2.1 of
            (Koh and Liang, 2017)<sup><a href="#koh_liang_2017">1</a></sup>
        Perturbation: Perturb a training point, see section 2.2 of
            (Koh and Liang, 2017)<sup><a href="#koh_liang_2017">1</a></sup>

    """

    Up = "up"
    Perturbation = "perturbation"


class Influence(Generic[TensorType], ABC):
    @property
    @abstractmethod
    def num_parameters(self):
        pass

    @property
    @abstractmethod

    def prepare_for_distributed(self) -> "Influence":
        """Overwrite this method, in case the instance has to be modified, before being distributed.
        Must be called explicitly.
        """
        return self

    @abstractmethod
    def up_weighting(
        self,
        z_test_factors: TensorType,
        x: TensorType,
        y: TensorType,
    ) -> TensorType:
        pass

    @abstractmethod
    def perturbation(
        self,
        z_test_factors: TensorType,
        x: TensorType,
        y: TensorType,
    ) -> TensorType:
        pass

    @abstractmethod
    def factors(self, x: TensorType, y: TensorType) -> InverseHvpResult[TensorType]:
        pass

    @abstractmethod
    def values(
        self,
        x_test: TensorType,
        y_test: TensorType,
        x: TensorType,
        y: TensorType,
        influence_type: InfluenceType,
    ) -> InverseHvpResult:
        pass

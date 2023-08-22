from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

TensorType = TypeVar("TensorType", bound=Sequence)
ModelType = TypeVar("ModelType", bound="TwiceDifferentiable")
DataLoaderType = TypeVar("DataLoaderType", bound=Iterable)


@dataclass(frozen=True)
class InverseHvpResult(Generic[TensorType]):
    x: TensorType
    info: Dict[str, Any]

    def __iter__(self):
        return iter((self.x, self.info))


class TwiceDifferentiable(ABC, Generic[TensorType]):
    """
    Wraps a differentiable model and loss and provides methods to compute gradients and
    second derivative of the loss wrt. the model parameters
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

    def grad(
        self, x: TensorType, y: TensorType, create_graph: bool = False
    ) -> TensorType:
        """
        Calculates gradient of model parameters wrt. the model parameters.

        :param x: A matrix representing the features $x_i$.
        :param y: A matrix representing the target values $y_i$.
            gradients. This is important for further differentiation on input
            parameters.
        :param create_graph:
        :return: A tuple where: the first element is an array with the
            gradients of the model, and the second element is the input to the
            model as a grad parameters. This can be used for further
            differentiation.
        """
        pass

    def hessian(self, x: TensorType, y: TensorType) -> TensorType:
        """Calculates the full Hessian of $L(f(x),y)$ with respect to the model
        parameters given data ($x$ and $y$).

        :param x: An array representing the features $x_i$.
        :param y: An array representing the target values $y_i$.
        :return: The hessian of the model, i.e. the second derivative wrt. the
            model parameters.
        """
        pass

    @staticmethod
    @abstractmethod
    def mvp(
        grad_xy: TensorType,
        v: TensorType,
        backprop_on: TensorType,
        *,
        progress: bool = False,
    ) -> TensorType:
        """
        Calculates second order derivative of the model along directions v.
        This second order derivative can be selected through the backprop_on argument.

        :param grad_xy: an array [P] holding the gradients of the model
            parameters wrt input $x$ and labels $y$, where P is the number of
            parameters of the model. It is typically obtained through
            self.grad.
        :param v: An array ([DxP] or even one dimensional [D]) which
            multiplies the matrix, where D is the number of directions.
        :param progress: True, iff progress shall be printed.
        :param backprop_on: tensor used in the second backpropagation (the first
            one is along $x$ and $y$ as defined via grad_xy).
        :returns: A matrix representing the implicit matrix vector product
            of the model along the given directions. Output shape is [DxP] if
            backprop_on is None, otherwise [DxM], with M the number of elements
            of backprop_on.
        """


class TensorUtilities(Generic[TensorType, ModelType], ABC):
    twice_differentiable_type: Type[TwiceDifferentiable]
    registry: Dict[Type[TwiceDifferentiable], Type["TensorUtilities"]] = {}

    def __init_subclass__(cls, **kwargs):
        """
        Automatically registers non-abstract subclasses in the registry.

        Checks if `twice_differentiable_type` is defined in the subclass and
        is of correct type. Raises `TypeError` if either attribute is missing or incorrect.

        :param kwargs: Additional keyword arguments.
        :raise TypeError: If the subclass does not define `twice_differentiable_type`,
        or if it is not of correct type.
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
    def einsum(equation, *operands) -> TensorType:
        """Sums the product of the elements of the input :attr:`operands` along dimensions specified using a notation
        based on the Einstein summation convention.
        """

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
        Factory method to create an instance of `TensorUtilities` from an instance of `TwiceDifferentiable`.

        :param twice_diff: An instance of `TwiceDifferentiable`
            for which a corresponding `TensorUtilities` object is required.
        :return: An instance of `TensorUtilities` corresponding to the provided `TwiceDifferentiable` object.
        :raises KeyError: If there's no registered `TensorUtilities` for the provided `TwiceDifferentiable` type.
        """
        tu = cls.registry.get(type(twice_diff), None)

        if tu is None:
            raise KeyError(
                f"No registered TensorUtilities for the type {type(twice_diff).__name__}"
            )

        return tu


class DataLoaderUtilities(Generic[DataLoaderType], ABC):

    data_loader_type: Type[DataLoaderType]
    registry: Dict[Type[DataLoaderType], Type["DataLoaderUtilities"]] = {}

    def __init_subclass__(cls, **kwargs):
        """
        Automatically registers subclasses in the registry.

        Checks if `data_loader_type` is defined in the subclass and
        is of correct type. Raises `TypeError` if either attribute is missing or incorrect.
        :param kwargs: Additional keyword arguments.
        :raise TypeError: If the subclass does not define `twice_differentiable_type`,
        or if it is not of correct type.
        """
        if not hasattr(cls, "data_loader_type") or not isinstance(
            cls.data_loader_type, type
        ):
            raise TypeError(f"'data_loader_type' must be a type object")

        cls.registry[cls.data_loader_type] = cls

        super().__init_subclass__(**kwargs)

    @staticmethod
    @abstractmethod
    def len_data(data: DataLoaderType) -> int:
        """Get the number of data points"""

    @staticmethod
    @abstractmethod
    def batch_size(data: DataLoaderType) -> int:
        """Get the number of batches for data"""

    @classmethod
    def from_data_loader(
        cls,
        data_loader: DataLoaderType,
    ) -> Type["DataLoaderUtilities"]:
        """
        Factory method to create an instance of `DataLoaderUtilities` from a given data loader.

        :param data_loader: An instance of data loader for which a corresponding
            `DataLoaderUtilities` object is required.
        :return: An instance of `DataLoaderUtilities` corresponding to the provided data loader.
        :raises KeyError: If there's no registered `DataLoaderUtilities` for the provided data loader type.
        """
        dl_u = cls.registry.get(type(data_loader), None)

        if dl_u is None:
            raise KeyError(
                f"No registered DataLoaderUtilities for the type {type(data_loader).__name__}"
            )

        return dl_u

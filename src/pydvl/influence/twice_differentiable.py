from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Sequence, Type, TypeVar

TensorType = TypeVar("TensorType", bound=Sequence)


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


class TensorUtilities(Generic[TensorType], ABC):
    twice_differentiable_type: Type[TwiceDifferentiable]
    registry: Dict[Type[TwiceDifferentiable], Type["TensorUtilities"]] = {}

    def __init_subclass__(cls, abstract: bool = False, **kwargs):
        """
        Automatically registers non-abstract subclasses in the registry.

        Checks if `twice_differentiable_type` is defined in the subclass and
        is of correct type. Raises `TypeError` if either attribute is missing or incorrect.

        :param abstract: If True, the subclass won't be registered. Default is False.
        :param kwargs: Additional keyword arguments.
        :raise TypeError: If the subclass does not define `twice_differentiable_type`,
        or if it is not of correct type.
        """
        if not abstract:
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
    def eye(dim: int, **kwargs) -> TensorType:
        """Identity tensor of dimension dim"""

    @classmethod
    def from_twice_differentiable(
        cls,
        twice_diff: TwiceDifferentiable,
    ) -> Type["TensorUtilities"]:
        tu = cls.registry.get(type(twice_diff), None)

        if tu is None:
            raise KeyError()

        return tu

from abc import ABC
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    NamedTuple,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

TensorType = TypeVar("TensorType", bound=Sequence)
ModelType = TypeVar("ModelType")
DeviceType = TypeVar("DeviceType")


class iHVPResult(NamedTuple):
    x: TensorType
    info: Dict[str, Any]


class TwiceDifferentiable(ABC, Generic[TensorType, ModelType, DeviceType]):
    """
    Wraps a differentiable model and loss and provides methods to compute the
    second derivative of the loss wrt. the model parameters.
    """

    def __init__(
        self,
        model: ModelType,
        loss: Callable[[TensorType, TensorType], TensorType],
        device: DeviceType,
    ):
        self.device = device
        pass

    @property
    def num_params(self) -> int:
        """Returns the number of parameters of the model"""
        pass

    @property
    def parameters(self) -> List[TensorType]:
        """Returns all the model parameters that require differentiation"""
        pass

    def split_grad(
        self, x: TensorType, y: TensorType, *, progress: bool = False
    ) -> TensorType:
        """
        Calculate the gradient of the model wrt each input x and labels y.
        The output is therefore of size [Nxp], with N the amount of points (the
        length of x and y) and P the number of parameters.

        :param x: An array representing the features $x_i$.
        :param y: An array representing the predicted target values $y_i$.
        :param progress: ``True`` to display progress.
        :return: An array representing the gradients wrt. the parameters of the
            model.
        """
        pass

    def grad(
        self, x: TensorType, y: TensorType, *, x_requires_grad: bool = False
    ) -> Tuple[TensorType, TensorType]:
        """
        Calculates gradient of model parameters wrt. the model parameters.

        :param x: A matrix representing the features $x_i$.
        :param y: A matrix representing the target values $y_i$.
        :param x_requires_grad: If True, the input $x$ is marked as requiring
            gradients. This is important for further differentiation on input
            parameters.
        :return: A tuple where: the first element is an array with the
            gradients of the model, and the second element is the input to the
            model as a grad parameters. This can be used for further
            differentiation.
        """
        pass

    def hessian(
        self, x: TensorType, y: TensorType, *, progress: bool = False
    ) -> TensorType:
        """Calculates the full Hessian of $L(f(x),y)$ with respect to the model
        parameters given data ($x$ and $y$).

        :param x: An array representing the features $x_i$.
        :param y: An array representing the target values $y_i$.
        :param progress: ``True`` to display progress.
        :return: The hessian of the model, i.e. the second derivative wrt. the
            model parameters.
        """
        pass

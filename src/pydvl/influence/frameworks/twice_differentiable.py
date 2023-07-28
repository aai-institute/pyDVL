from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Sequence, Tuple, TypeVar

TensorType = TypeVar("TensorType", bound=Sequence)
ModelType = TypeVar("ModelType")
DeviceType = TypeVar("DeviceType")


@dataclass(frozen=True)
class InverseHvpResult(Generic[TensorType]):
    x: TensorType
    info: Dict[str, Any]

    def __iter__(self):
        return iter((self.x, self.info))


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

    def grad(
        self, x: TensorType, y: TensorType, create_graph: bool = True
    ) -> TensorType:
        """
        Calculates gradient of model parameters wrt. the model parameters.

        :param x: A matrix representing the features $x_i$.
        :param y: A matrix representing the target values $y_i$.
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

from abc import ABC
from typing import Callable, Generic, List, Sequence, Tuple, TypeVar

TensorType = TypeVar("TensorType", bound=Sequence)
ModelType = TypeVar("ModelType")


class TwiceDifferentiable(ABC, Generic[TensorType, ModelType]):
    def __init__(
        self,
        model: ModelType,
        loss: Callable[[TensorType, TensorType], TensorType],
    ):
        pass

    def num_params(self) -> int:
        """Returns the number of parameters of the model"""
        pass

    def parameters(self) -> List[TensorType]:
        """Returns all the model parameters that require differentiation"""
        pass

    def split_grad(
        self,
        x: TensorType,
        y: TensorType,
        progress: bool = False,
    ) -> TensorType:
        """
        Calculate the gradient of the model wrt each input x and labels y.
        The output is therefore of size [Nxp], with N the amount of points (the
        length of x and y) and P the number of parameters.
        """
        pass

    def grad(self, x: TensorType, y: TensorType) -> Tuple[TensorType, TensorType]:
        """
        Calculates the gradient of model parameters with respect to input x
        and labels y.
        """
        pass

    def hessian(
        self,
        x: TensorType,
        y: TensorType,
        progress: bool = False,
    ) -> TensorType:
        """Calculates the explicit hessian of model parameters given data ($x$
        and $y$)."""
        pass

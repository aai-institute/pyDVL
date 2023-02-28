from abc import ABC
from typing import Generic, Optional, Tuple, TypeVar

from numpy.typing import NDArray

TensorType = TypeVar("TensorType")


class BaseTwiceDifferentiable(ABC, Generic[TensorType]):
    def num_params(self) -> int:
        pass

    def split_grad(
        self, x: TensorType, y: TensorType, progress: bool = False
    ) -> NDArray:
        """
        Calculate the gradient of the model wrt each input x and labels y.
        The output is therefore of size [Nxp], with N the amout of points (the
        length of x and y) and P the number of parameters.
        """
        pass

    def grad(self, x: TensorType, y: TensorType) -> Tuple[NDArray, TensorType]:
        """
        It calculates the gradient of model parameters with respect to input x
        and labels y.
        """
        pass

    def mvp(
        self,
        grad_xy: TensorType,
        v: TensorType,
        progress: bool = False,
        backprop_on: Optional[TensorType] = None,
    ) -> NDArray:
        """
        Calculates second order derivative of the model along directions v,
        which can be a single vector or a matrix (thus the method must support
        broadcasting). This second order derivative can be on the model parameters or on
        another input parameter, selected via the backprop_on argument.
        """
        pass

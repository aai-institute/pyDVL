from abc import ABC
from typing import Optional, Tuple, Union

try:
    import torch

    _TORCH_INSTALLED = True
except ImportError:
    _TORCH_INSTALLED = False

from numpy.typing import NDArray

__all__ = ["TwiceDifferentiable", "TensorType"]

TensorType = Union[NDArray, "torch.Tensor"]


class TwiceDifferentiable(ABC):
    def num_params(self) -> int:
        pass

    def split_grad(
        self, x: TensorType, y: TensorType, progress: bool = False
    ) -> NDArray:
        """
        Calculate the gradient of the model wrt each input x and labels y.
        The output is therefore of size [Nxp], with N the amout of points (the length of x and y) and
        P the number of parameters.
        """
        pass

    def grad(self, x: TensorType, y: TensorType) -> Tuple[NDArray, TensorType]:
        """
        It calculates the gradient of model parameters with respect to input x and labels y.
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
        Calculate the hessian vector product over the loss with all input parameters x and y with the vector v.
        """
        pass

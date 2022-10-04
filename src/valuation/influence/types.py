from abc import ABC
from typing import Callable, Iterable, Optional, Tuple

from numpy import ndarray

__all__ = [
    "TwiceDifferentiable",
    "MatrixVectorProduct",
    "MatrixVectorProductInversionAlgorithm",
]


class TwiceDifferentiable(ABC):
    def num_params(self) -> int:
        pass

    def split_grad(self, x: ndarray, y: ndarray, progress: bool = False) -> ndarray:
        """
        Calculate the gradient of the model wrt each input x and labels y.
        The output is therefore of size [Nxp], with N the amout of points (the length of x and y) and
        P the number of parameters.
        """
        pass

    def grad(self, x: ndarray, y: ndarray) -> Tuple[ndarray, ndarray]:
        """
        It calculates the gradient of model parameters with respect to input x and labels y.
        """
        pass

    def mvp(
        self,
        grad_xy: ndarray,
        v: ndarray,
        progress: bool = False,
        backprop_on: Optional[Iterable] = None,
    ) -> ndarray:
        """
        Calculate the hessian vector product over the loss with all input parameters x and y with the vector v.
        """
        pass


MatrixVectorProduct = Callable[[ndarray], ndarray]

MatrixVectorProductInversionAlgorithm = Callable[
    [MatrixVectorProduct, ndarray], ndarray
]

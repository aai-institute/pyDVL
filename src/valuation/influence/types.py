from typing import Callable, Protocol

from numpy import ndarray

__all__ = [
    "TwiceDifferentiable",
    "MatrixVectorProduct",
    "MatrixVectorProductInversionAlgorithm",
]


class TwiceDifferentiable(Protocol):
    def num_params(self) -> int:
        pass

    def grad(self, x: ndarray, y: ndarray, progress: bool = False) -> ndarray:
        """
        Calculate the gradient with respect to the parameters of the module with input parameters x[i] and y[i].
        """
        pass

    def mvp(
        self,
        x: ndarray,
        y: ndarray,
        v: ndarray,
        progress: bool = False,
        second_x: bool = False,
        **kwargs
    ) -> ndarray:
        """
        Calculate the hessian vector product over the loss with all input parameters x and y with the vector v.
        """
        pass


MatrixVectorProduct = Callable[[ndarray], ndarray]

MatrixVectorProductInversionAlgorithm = Callable[
    [MatrixVectorProduct, ndarray], ndarray
]

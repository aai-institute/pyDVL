from abc import abstractmethod

import numpy as np


class TwiceDifferentiable:
    @abstractmethod
    def grad(self, x: np.ndarray, y: np.ndarray, progress: bool = False) -> np.ndarray:
        """
        Calculate the gradient with respect to the parameters of the module with input parameters x[i] and y[i].
        """
        pass

    @abstractmethod
    def hvp(
        self, x: np.ndarray, y: np.ndarray, v: np.ndarray, progress: bool = False
    ) -> np.ndarray:
        """
        Calculate the hessian vector product over the loss with all input parameters x and y with the vector v.
        """
        pass

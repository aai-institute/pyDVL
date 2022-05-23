from abc import abstractmethod
from functools import partial

import numpy as np

from valuation.utils.algorithms import conjugate_gradient


class NaiveInverseHessianVectorProductMixing:
    def inverse_hvp(self, theta: np.ndarray, x: np.ndarray) -> np.darray:
        """
        Calculate the inverse hessian vector product by inverting the Hessian H(theta) and
        multiplying it with the input value x.

        :param theta: A np.ndarray of type float with shape [num_parameters].
        :param x: A np.ndarray of type float with shape [num_parameters] or [num_samples, num_parameters].
        :returns A np.ndarray of type float with the same shape as x.
        """
        hessian = self.hessian(theta)
        inv_hessian = np.linalg.inv(hessian)
        return x @ inv_hessian.T

    @abstractmethod
    def hessian(self, theta: np.ndarray) -> np.ndarray:
        pass


class ConjugateGradientInverseHessianVectorProductMixing:
    def inverse_hvp(self, theta: np.ndarray, x: np.ndarray) -> np.darray:
        """
        Calculate the inverse hessian vector product by

        :param theta: A np.ndarray of type float with shape [num_parameters].
        :param x: A np.ndarray of type float with shape [num_parameters] or [num_samples, num_parameters].
        :returns A np.ndarray of type float with the same shape as x.
        """
        hvp = partial(self.hvp, self, theta)
        return conjugate_gradient(hvp, x)

    @abstractmethod
    def hvp(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        pass

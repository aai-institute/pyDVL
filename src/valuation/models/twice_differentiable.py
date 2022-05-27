from abc import abstractmethod

import numpy as np


class TwiceDifferentiable:

    @abstractmethod
    def grad(self, x: np.ndarray, y: np.ndarray, progress: bool = False) -> np.ndarray:
        pass

    @abstractmethod
    def hvp(self, x: np.ndarray, y: np.ndarray, v: np.ndarray, progress: bool = False) -> np.ndarray:
        pass
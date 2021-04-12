from typing import Protocol
from numpy import ndarray


class SupervisedModel(Protocol):
    """ Pedantic: only here for the type hints. """
    def fit(self, x: ndarray, y: ndarray):
        pass

    def predict(self, x: ndarray) -> ndarray:
        pass

    def score(self, x: ndarray, y: ndarray) -> float:
        pass

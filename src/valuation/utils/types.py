from typing import Callable, Protocol, Type, Union

from numpy import ndarray

__all__ = [
    "SupervisedModel",
    "Scorer",
]


class SupervisedModel(Protocol):
    """Pedantic: only here for the type hints."""

    def fit(self, x: ndarray, y: ndarray):
        pass

    def predict(self, x: ndarray) -> ndarray:
        pass

    def score(self, x: ndarray, y: ndarray) -> float:
        pass


# ScorerNames = Literal[very long list here]
# instead... ScorerNames = str
Scorer = Union[str, Callable[[SupervisedModel, ndarray, ndarray], float]]

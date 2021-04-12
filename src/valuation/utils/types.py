from typing import Callable, Protocol, TypeVar, Union
from numpy import ndarray


class SupervisedModel(Protocol):
    """ Pedantic: only here for the type hints. """
    def fit(self, x: ndarray, y: ndarray):
        pass

    def predict(self, x: ndarray) -> ndarray:
        pass

    def score(self, x: ndarray, y: ndarray) -> float:
        pass


# ScorerNames = Literal[very long list here]
# instead... ScorerNames = str

Scorer = TypeVar('Scorer',
                 str, Callable[[SupervisedModel, ndarray, ndarray], float])

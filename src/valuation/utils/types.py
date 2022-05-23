from abc import abstractmethod
from typing import Callable, Protocol, TypeVar
from numpy import ndarray
import numpy as np

__all__ = ["SupervisedModel", "Scorer", "unpackable"]


class SupervisedModel(Protocol):

    """Pedantic: only here for the type hints."""

    def fit(self, x: ndarray, y: ndarray):
        pass

    def predict(self, x: ndarray) -> ndarray:
        pass

    def score(self, x: ndarray, y: ndarray) -> float:
        pass

    @property
    @abstractmethod
    def parameters(self) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_hvp(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Calculate the inverse hessian vector product v based on a vector x such that.
        H(theta)^{-1} x. Whereas H is the Hessian matrix around parameters theta.

        :param theta: A np.ndarray of type float with shape [num_parameters].
        :param x: A np.ndarray of type float with shape [num_parameters] or [num_samples, num_parameters].
        :returns A np.ndarray of type float with the same shape as x.
        """
        pass


# ScorerNames = Literal[very long list here]
# instead... ScorerNames = str

Scorer = TypeVar("Scorer", str, Callable[[SupervisedModel, ndarray, ndarray], float])


def unpackable(cls: type) -> type:
    """A class decorator that allows unpacking of all attributes of an object
    with the double asterisk operator. E.g.

       @unpackable
       @dataclass
       class Schtuff:
           a: int
           b: str

       x = Schtuff(a=1, b='meh')
       d = dict(**x)
    """

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return getattr(self, item)

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        for k in self.keys():
            yield getattr(self, k)

    # HACK: I needed this somewhere else
    def update(self, values: dict):
        for k, v in values.items():
            setattr(self, k, v)

    def items(self):
        for k in self.keys():
            yield k, getattr(self, k)

    setattr(cls, "keys", keys)
    setattr(cls, "__getitem__", __getitem__)
    setattr(cls, "__len__", __len__)
    setattr(cls, "__iter__", __iter__)
    setattr(cls, "update", update)
    setattr(cls, "items", items)

    return cls

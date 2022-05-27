from typing import Callable, Protocol, TypeVar, NewType

import torch
from numpy import ndarray
import numpy as np

__all__ = ['SupervisedModel', 'Scorer', 'unpackable']


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


class TorchObjective(Protocol):

    def __call__(self, x: torch.tensor, y: torch.tensor, **kwargs) -> torch.tensor:
        pass


class BatchInfluenceFunction(Protocol):

    def __call__(self, v: np.ndarray) -> np.ndarray:
        pass


def unpackable(cls: type) -> type:
    """ A class decorator that allows unpacking of all attributes of an object
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

    setattr(cls, 'keys', keys)
    setattr(cls, '__getitem__', __getitem__)
    setattr(cls, '__len__', __len__)
    setattr(cls, '__iter__', __iter__)
    setattr(cls, 'update', update)
    setattr(cls, 'items', items)

    return cls

import inspect
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


Scorer = Callable[[SupervisedModel, ndarray, ndarray], float]


def unpackable(cls: Type) -> Type:
    """A class decorator that allows unpacking of all attributes of an object
    with the double asterisk operator.

    :Example:

    >>> @unpackable
    ... @dataclass
    ... class Schtuff:
    ...    a: int
    ...    b: str
    >>> x = Schtuff(a=1, b='meh')
    >>> d = dict(**x)
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


def maybe_add_argument(fun: Callable, new_arg: str):
    """Wraps a function to accept (and ignore) the given named parameter"""
    params = inspect.signature(fun).parameters
    if new_arg in params.keys():
        return fun

    def wrapper(*args, **kwargs):
        try:
            del kwargs[new_arg]
        except KeyError:
            pass
        return fun(*args, **kwargs)

    return wrapper

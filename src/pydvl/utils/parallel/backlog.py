from copy import copy
from functools import partial
from typing import Callable, Generator, Generic, List, Tuple, TypeVar

T = TypeVar("T")


def wrap_index(*args, i: int, fn: Callable, **kwargs):
    return i, fn(*args, **kwargs)


class Backlog(Generic[T]):
    """A backlog is a queue of items added in no particular order. Each item has an
    index used to determine the sequence in which the items are processed. A function
    call can be modified using the wrap method to include this index in the output.
    This modification should be applied before invoking the function."""

    def __init__(self):
        self._backlog: List[Tuple[int, T]] = []
        self._n_delivered = 0
        self._n_registered = 0
        self._n_wrapped = 0

    def add(self, item: Tuple[int, T]):
        self._backlog.append(item)
        self._backlog = sorted(self._backlog, key=lambda t: t[0])
        self._n_registered += 1

    def get(self) -> Generator[T, None, None]:
        while len(self._backlog) > 0 and self._backlog[0][0] == self._n_delivered:
            self._n_delivered += 1
            yield self._backlog[0][1]
            self._backlog = self._backlog[1:]

    def wrap(self, fn: Callable) -> Callable:
        self._n_wrapped += 1
        return partial(wrap_index, fn=fn, i=copy(self._n_wrapped - 1))

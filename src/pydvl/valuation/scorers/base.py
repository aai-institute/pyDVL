"""
This module implements the base class for all scorers used by valuation methods.
"""

from abc import ABC, abstractmethod


class Scorer(ABC):
    """A scoring callable that takes a model and returns a scalar.

    !!! tip "Added in version 0.10.0"
        ABC added
    """

    default: float
    name: str
    range: tuple[float, float]

    @abstractmethod
    def __call__(self, model) -> float: ...

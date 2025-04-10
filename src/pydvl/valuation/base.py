"""
This module declares the abstract base classes for all valuation methods.
A **valuation method** is any function that computes a value for each data point in a
dataset.

!!! info
    For information on data valuation, read [the introduction][data-valuation-intro].
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterable

from deprecate import deprecated
from typing_extensions import Self

from pydvl.utils.exceptions import NotFittedException
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult

__all__ = ["Valuation", "ModelFreeValuation"]

logger = logging.getLogger(__name__)


class Valuation(ABC):
    """Abstract base class for all valuation methods."""

    algorithm_name: str = "Valuation"

    def __init__(self) -> None:
        self._result: ValuationResult | None = None

    @abstractmethod
    def fit(
        self, data: Dataset, continue_from: ValuationResult | None = None
    ) -> Self: ...

    def _init_or_check_result(
        self, data: Dataset, result: ValuationResult | None = None
    ) -> ValuationResult:
        """Initialize the valuation result or check that a previously restored one
        (with `load_result()`) matches the data.

        Args:
            data: The dataset to use for initialization or checking
            result: The previously loaded valuation result to check against the data.
        Returns:
            A zeroed valuation result, or a previously loaded one if it matches the data.
        Raises:
            ValueError: If the dataset does not match the loaded result.
        """

        if result is None:
            return ValuationResult.zeros(
                algorithm=str(self), indices=data.indices, data_names=data.names
            )

        try:
            assert all(result.indices == data.indices)
            assert all(result.names == data.names)
        except (AssertionError, ValueError) as e:
            raise ValueError(
                "Either the indices or the names of the dataset do not match those of "
                "the passed valuation result."
            ) from e

        return result

    @property
    def result(self) -> ValuationResult:
        """The current valuation result (not a copy)."""
        if not self.is_fitted:
            raise NotFittedException(type(self))
        assert self._result is not None

        return self._result

    @deprecated(
        target=None,
        deprecated_in="0.10.0",
        remove_in="0.11.0",
    )
    def values(self, sort: bool = False) -> ValuationResult:
        """Returns a copy of the valuation result.

        The valuation must have been run with `fit()` before calling this method.

        Args:
            sort: Whether to sort the valuation result by value before returning it.
        Returns:
            The result of the valuation.
        """
        if not self.is_fitted:
            raise NotFittedException(type(self))
        assert self._result is not None

        r = self._result.copy()
        if sort:
            r.sort(inplace=True)
        return r

    @property
    def is_fitted(self) -> bool:
        return self._result is not None

    def __str__(self):
        return getattr(self, "algorithm_name", self.__class__.__name__)


class ModelFreeValuation(Valuation, ABC):
    """
    TODO: Just a stub, probably should not inherit from Valuation.
    """

    def __init__(self, references: Iterable[Dataset]):
        super().__init__()
        self.datasets = references

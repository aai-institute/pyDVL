from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult

__all__ = ["Valuation", "ModelFreeValuation"]


class Valuation(ABC):
    def __init__(self):
        self.result: ValuationResult | None = None

    @abstractmethod
    def fit(self, data: Dataset) -> Valuation:
        ...

    def values(self, sort: bool) -> ValuationResult:
        """Returns the valuation result.

        The valuation must have been run with `fit()` before calling this method.

        Returns:
            The result of the valuation.
        """
        if not self.is_fitted:
            raise RuntimeError("Valuation is not fitted")
        assert self.result is not None

        if sort:
            self.result.sort()
        return self.result

    @property
    def is_fitted(self) -> bool:
        return self.result is not None


class ModelFreeValuation(Valuation, ABC):
    """
    TODO: Just a stub
    """

    def __init__(self, references: Iterable[Dataset]):
        super().__init__()
        self.datasets = references

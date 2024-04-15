from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic

from pydvl.valuation.dataset import Dataset
from pydvl.valuation.types import SampleT


class UtilityBase(Generic[SampleT], ABC):
    training_data: Dataset | None

    @abstractmethod
    def __call__(self, sample: SampleT) -> float:
        ...

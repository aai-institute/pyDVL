from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Generic

from typing_extensions import Self

from pydvl.valuation.dataset import Dataset
from pydvl.valuation.types import SampleT


class UtilityBase(Generic[SampleT], ABC):
    training_data: Dataset | None

    def with_dataset(self, data: Dataset) -> Self:
        copied = copy.copy(self)
        copied.training_data = data
        return copied

    @abstractmethod
    def __call__(self, sample: SampleT) -> float:
        ...

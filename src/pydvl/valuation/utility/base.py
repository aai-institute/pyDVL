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
    def __call__(self, sample: SampleT | None) -> float:
        """
        !!! Note
            Calls with empty samples or None must always return the same valid value,
            e.g. 0, or whatever makes sense for the utility. Some samplers (e.g.
            permutations) depend on this.

        Args:
            sample:

        Returns:
            The evaluation of the utility for the sample
        """
        ...

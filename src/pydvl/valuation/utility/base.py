"""
This module defines the base class for all utilities.
"""

from __future__ import annotations

import copy as cp
from abc import ABC, abstractmethod
from typing import Generic

from typing_extensions import Self

from pydvl.valuation.dataset import Dataset
from pydvl.valuation.types import SampleT


class UtilityBase(Generic[SampleT], ABC):
    """Base class for all utilities.

    A utility is a scalar-valued set function which will be evaluated over subsets of
    the training set.
    """

    _training_data: Dataset | None

    @property
    def training_data(self) -> Dataset | None:
        """Retrieves the training data used by this utility.

        This property is read-only. In order to set it, use
        [with_dataset()][pydvl.valuation.utility.base.UtilityBase.with_dataset].
        """
        return getattr(self, "_training_data", None)

    def with_dataset(self, data: Dataset, copy: bool = True) -> Self:
        """Returns the utility, or a copy of it, with the given dataset.
        Args:
            data: The dataset to use for utility fitting (training data)
            copy: Whether to copy the utility object or not. Valuation methods should
                always make copies to avoid unexpected side effects.
        Returns:
            The utility object.
        """
        utility = cp.copy(self) if copy else self
        utility._training_data = data
        return utility

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

    def __str__(self):
        """Returns a string representation of the utility.
        Subclasses should override this method to provide a more informative string
        """
        return f"{self.__class__.__name__}"

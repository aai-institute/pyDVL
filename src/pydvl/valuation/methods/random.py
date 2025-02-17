"""
This module implements a trivial random valuation method.

"""

import numpy as np
from typing_extensions import Self

from pydvl.utils import Seed
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult


class RandomValuation(Valuation):
    """
    A trivial valuation method that assigns random values to each data point.

    Values are in the range [0, 1), as generated by
    [ValuationResult.from_random][pydvl.valuation.result.ValuationResult.from_random].

    Successive calls to [fit()][pydvl.valuation.base.Valuation.fit] will generate
    different values.
    """

    def __init__(self, random_state: Seed):
        super().__init__()
        self.random_state = np.random.default_rng(random_state)

    def fit(self, train: Dataset) -> Self:
        """ Dummy fitting that generates a set of random values.

        Successive calls will generate different values.

        Args:
            train: used to determine the size of the valuation result
        Returns:
            self
        """
        self.result = ValuationResult.from_random(
            size=len(train), seed=self.random_state
        )
        return self

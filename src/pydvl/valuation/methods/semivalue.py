r"""
This module contains the base class for all semi-value valuation methods.

A **semi-value** is any valuation function with the form:

$$
v_\text{semi}(i) = \sum_{i=1}^n w(k)
                     \sum_{S \subset D_{-i}^{(k)}} [U(S_{+i})-U(S)],
$$

where $U$ is the utility, and the coefficients $w(k)$ satisfy the property:

$$
\sum_{k=1}^n w(k) = 1.
$$

This is the largest class of marginal-contribution-based valuation methods. These
compute the value of a data point by evaluating the change in utility when the data
point is removed from one or more subsets of the data.
"""

from __future__ import annotations

from abc import abstractmethod

from joblib import Parallel, delayed
from tqdm import tqdm

from pydvl.valuation.base import (
    Valuation,
    ensure_backend_has_generator_return,
    make_parallel_flag,
)
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers import IndexSampler
from pydvl.valuation.stopping import StoppingCriterion
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["SemivalueValuation"]


class SemivalueValuation(Valuation):
    r"""Abstract class to define semi-values.

    Implementations must only provide the `coefficient()` method, corresponding
    to the semi-value coefficient.

    !!! Note
        For implementation consistency, we slightly depart from the common definition
        of semi-values, which includes a factor $1/n$ in the sum over subsets.
        Instead, we subsume this factor into the coefficient $w(k)$.
        TODO: see ...

    Args:
        utility: Object to compute utilities.
        sampler: Sampling scheme to use.
        is_done: Stopping criterion to use.
        progress: Whether to show a progress bar.
    """

    algorithm_name = "Semi-Value"

    def __init__(
        self,
        utility: UtilityBase,
        sampler: IndexSampler,
        is_done: StoppingCriterion,
        progress: bool = False,
    ):
        super().__init__()
        self.utility = utility
        self.sampler = sampler
        self.is_done = is_done
        self.progress = progress

    @abstractmethod
    def coefficient(self, n: int, k: int) -> float:
        """Computes the coefficient for a given subset size.

        Args:
            n: Total number of elements in the set.
            k: Size of the subset for which the coefficient is being computed
        """
        ...

    def fit(self, data: Dataset):
        self.result = ValuationResult.zeros(
            # TODO: automate str representation for all Valuations (and find something better)
            algorithm=f"{self.__class__.__name__}-{self.utility.__class__.__name__}-{self.sampler.__class__.__name__}-{self.utility.model}-{self.is_done}",
            indices=data.indices,
            data_names=data.data_names,
        )

        ensure_backend_has_generator_return()
        flag = make_parallel_flag()

        self.utility.training_data = data

        parallel = Parallel(return_as="generator_unordered")
        strategy = self.sampler.make_strategy(self.utility, self.coefficient)
        processor = delayed(strategy.process)

        delayed_evals = parallel(
            processor(batch=list(batch), is_interrupted=flag)
            for batch in self.sampler.from_data(data)
        )
        for batch in tqdm(iterable=delayed_evals, disable=not self.progress):
            for evaluation in batch:
                self.result.update(evaluation.idx, evaluation.update)
                if self.is_done(self.result):
                    flag.set()
                    self.sampler.interrupt()
                    break

        #####################

        # FIXME: remove NaN checking after fit()?
        import logging

        import numpy as np

        logger = logging.getLogger(__name__)
        nans = np.isnan(self.result.values).sum()
        if nans > 0:
            logger.warning(
                f"{nans} NaN values in current result. "
                "Consider setting a default value for the Scorer"
            )

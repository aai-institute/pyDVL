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
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from typing_extensions import Self

from pydvl.utils.progress import Progress
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers import IndexSampler
from pydvl.valuation.stopping import StoppingCriterion
from pydvl.valuation.utility.base import UtilityBase
from pydvl.valuation.utils import (
    ensure_backend_has_generator_return,
    make_parallel_flag,
)

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
        skip_converged: Whether to skip converged indices. Convergence is determined
            by the stopping criterion's `converged` array.
        progress: Whether to show a progress bar.
    """

    algorithm_name = "Semi-Value"

    def __init__(
        self,
        utility: UtilityBase,
        sampler: IndexSampler,
        is_done: StoppingCriterion,
        skip_converged: bool = False,
        progress: dict[str, Any] | bool = False,
    ):
        super().__init__()
        self.utility = utility
        self.sampler = sampler
        self.is_done = is_done
        self.skip_converged = skip_converged
        self.tqdm_args: dict[str, Any] = {
            "desc": f"{self.__class__.__name__}: {str(is_done)}"
        }
        # HACK: parse additional args for the progress bar if any (we probably want
        #  something better)
        if isinstance(progress, bool):
            self.tqdm_args.update({"disable": not progress})
        else:
            self.tqdm_args.update(progress if isinstance(progress, dict) else {})

    @abstractmethod
    def coefficient(self, n: int, k: int, weight: float) -> float:
        """Returns the function computing the final coefficient to be used in the
        semi-value valuation.

        The semi-value coefficient is a function of the number of elements in the set,
        and the size of the subset for which the coefficient is being computed.
        Coefficients can be very large or very small, so that simply multiplying them
        with the rest of the factors in a semi-value computation can lead to overflow or
        underflow. To avoid this, we pass the other factors to this method, and delegate
        the choice of whether to multiply or divide to the implementation.

        Args:
            n: Total number of elements in the set.
            k: Size of the subset for which the coefficient is being computed
            weight: The weight coming from the samplers
        """
        ...

    def fit(self, data: Dataset) -> Self:
        self.result = ValuationResult.zeros(
            # TODO: automate str representation for all Valuations (and find something better)
            algorithm=f"{self.__class__.__name__}-{self.utility.__class__.__name__}-{self.sampler.__class__.__name__}-{self.is_done}",
            indices=data.indices,
            data_names=data.names,
        )
        ensure_backend_has_generator_return()

        self.is_done.reset()
        self.utility.training_data = data

        strategy = self.sampler.make_strategy(self.utility, self.coefficient)
        processor = delayed(strategy.process)
        updater = self.sampler.result_updater(self.result)

        with Parallel(return_as="generator_unordered") as parallel:
            with make_parallel_flag() as flag:
                delayed_evals = parallel(
                    processor(batch=list(batch), is_interrupted=flag)
                    for batch in self.sampler.generate_batches(data.indices)
                )
                for batch in Progress(delayed_evals, self.is_done, **self.tqdm_args):
                    for update in batch:
                        self.result = updater(update)
                        if self.skip_converged:
                            self.sampler.skip_indices = np.where(
                                self.is_done.converged
                            )[0]
                        if self.is_done(self.result):
                            flag.set()
                            self.sampler.interrupt()
                            return self
        return self

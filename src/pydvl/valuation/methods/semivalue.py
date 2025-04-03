r"""
This module contains the base class for all semi-value valuation methods.

A **semi-value** is any marginal contribution-based valuation method which weights
the marginal contributions of a data point $i$ to the utility of a subset $S$ by
weights $w(k)$, where $k$ is the size of the subset, fulfilling certain conditions. For
details, please refer to the [introduction to semi-values][semi-values-intro].

## Implementing new methods with importance sampling

!!! info "Semi-values and importance sampling"
    For a more detailed analysis of the ideas in this and the following section, please
    read [Sampling strategies for semi-values][semi-values-sampling].

Because almost every method employs Monte Carlo sampling of subsets, our architecture
allows for importance sampling. Early valuation methods chose samplers to
implicitly provide the weights $w(k)$ as exactly the sampling probabilities of sets
$p(S|k)$, e.g. [permutation Shapley][permutation-shapley-intro].

However, this is not a requirement. In fact, other methods employ different forms of
importance sampling as a means to reduce the variance both of the Monte Carlo estimates
and the utility function.

For this reason, our implementation allows mix-and-matching of any semi-value coefficient
with any sampler. For importance sampling, the mechanism is as follows:

* Choose a sampler to go with the semi-value. The sampler must implement the
  `log_weight()` property, which returns the logarithm of the sampling probability of a
  subset $S$ of size $k$, i.e. $p(S|k).$ Note that this is **not** p(|S|=k).$ The sampler
  also implements an [EvaluationStrategy][pydvl.valuation.samplers.base.EvaluationStrategy]
  which is used to compute the utility of the sampled subsets in subprocesses.

* Subclass [SemivalueValuation][pydvl.valuation.methods.semivalue.SemivalueValuation]
  and implement the `log_coefficient()` method. This method should return the final
  coefficient in log-space, i.e. the natural logarithm of the coefficient, for numerical
  stability. The coefficient is a function of the number of elements in the set $n$ and
  the size of the subset $k$ for which the coefficient is being computed, and of the
  sampler's weight. You can combine the method's coefficient and the weight in any way.
  For instance, in order to entirely compensate for the sampling distribution one simply
  subtracts the log-weights from the log-coefficient.

## Disabling importance sampling

In case you have a sampler that already provides the coefficients you need implicitly
as the sampling probabilities, you can override the `log_coefficient` property to
return `None`.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from typing_extensions import Self

from pydvl.utils.functional import suppress_warnings
from pydvl.utils.progress import Progress
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.parallel import (
    ensure_backend_has_generator_return,
    make_parallel_flag,
)
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers import IndexSampler
from pydvl.valuation.stopping import StoppingCriterion
from pydvl.valuation.types import SemivalueCoefficient
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["SemivalueValuation"]


logger = logging.getLogger(__name__)


class SemivalueValuation(Valuation):
    """Abstract class to define semi-values.

    Implementations must only provide the `log_coefficient()` property, corresponding
    to the semi-value coefficient.

    !!! Note
        For implementation consistency, we slightly depart from the common definition
        of semi-values, which includes a factor $1/n$ in the sum over subsets.
        Instead, we subsume this factor into the coefficient $w(k)$.

    Args:
        utility: Object to compute utilities.
        sampler: Sampling scheme to use.
        is_done: Stopping criterion to use.
        skip_converged: Whether to skip converged indices, as determined by the
            stopping criterion's `converged` array.
        show_warnings: Whether to show warnings.
        progress: Whether to show a progress bar. If a dictionary, it is passed to
            `tqdm` as keyword arguments, and the progress bar is displayed.
    """

    algorithm_name = "Semi-Value"

    def __init__(
        self,
        utility: UtilityBase,
        sampler: IndexSampler,
        is_done: StoppingCriterion,
        skip_converged: bool = False,
        show_warnings: bool = True,
        progress: dict[str, Any] | bool = False,
    ):
        super().__init__()
        self.utility = utility
        self.sampler = sampler
        self.is_done = is_done
        self.skip_converged = skip_converged
        if skip_converged:  # test whether the sampler supports skipping indices:
            self.sampler.skip_indices = np.array([], dtype=np.int_)
        self.show_warnings = show_warnings
        self.tqdm_args: dict[str, Any] = {"desc": str(self)}
        # HACK: parse additional args for the progress bar if any (we probably want
        #  something better)
        if isinstance(progress, bool):
            self.tqdm_args.update({"disable": not progress})
        elif isinstance(progress, dict):
            self.tqdm_args.update(progress)
        else:
            raise TypeError(f"Invalid type for progress: {type(progress)}")

    # TODO: automate str representation for all Valuations (and find something better)
    def __str__(self):
        return (
            f"{self.__class__.__name__}-{self.utility.__class__.__name__}-"
            f"{self.sampler.__class__.__name__}-{self.is_done}"
        )

    @property
    @abstractmethod
    def log_coefficient(self) -> SemivalueCoefficient | None:
        """This property returns the function computing the semi-value coefficient.

        Return `None` in subclasses that do not need to correct for the sampling
        distribution probabilities because of a specific, fixed sampler choice which
        already yields the semi-value coefficient.
        """
        ...

    @suppress_warnings(flag="show_warnings")
    def fit(self, data: Dataset) -> Self:
        self.result = ValuationResult.zeros(
            algorithm=str(self),
            indices=data.indices,
            data_names=data.names,
        )
        ensure_backend_has_generator_return()

        self.is_done.reset()
        self.utility = self.utility.with_dataset(data)

        strategy = self.sampler.make_strategy(self.utility, self.log_coefficient)
        updater = self.sampler.result_updater(self.result)
        processor = delayed(strategy.process)

        with Parallel(return_as="generator_unordered") as parallel:
            with make_parallel_flag() as flag:
                delayed_evals = parallel(
                    processor(batch=list(batch), is_interrupted=flag)
                    for batch in self.sampler.generate_batches(data.indices)
                )
                for batch in Progress(delayed_evals, self.is_done, **self.tqdm_args):
                    for update in batch:
                        self.result = updater.process(update)
                    if self.is_done(self.result):
                        flag.set()
                        self.sampler.interrupt()
                        break
                    if self.skip_converged:
                        self.sampler.skip_indices = data.indices[self.is_done.converged]
        logger.debug(f"Fitting done after {updater.n_updates} value updates.")
        return self

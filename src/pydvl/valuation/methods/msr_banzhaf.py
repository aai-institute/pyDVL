"""
This module implements the MSR-Banzhaf valuation method, as described in
(Wang et. al.)<sup><a href="wang_data_2023">3</a></sup>.

## References

[^1]: <a name="wang_data_2023"></a>Wang, J.T. and Jia, R., 2023.
    [Data Banzhaf: A Robust Data Valuation Framework for Machine Learning](https://proceedings.mlr.press/v206/wang23e.html).
    In: Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, pp. 6388-6421.

"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from typing_extensions import Self

from pydvl.utils.progress import Progress
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods.semivalue import SemivalueValuation
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers import MSRSampler
from pydvl.valuation.stopping import StoppingCriterion
from pydvl.valuation.types import ValueUpdateKind
from pydvl.valuation.utility.base import UtilityBase
from pydvl.valuation.utils import (
    ensure_backend_has_generator_return,
    make_parallel_flag,
)

__all__ = ["MSRBanzhafValuation"]


class MSRBanzhafValuation(SemivalueValuation):
    algorithm_name = "MSR-Banzhaf"

    def __init__(
        self,
        utility: UtilityBase,
        sampler: MSRSampler,
        is_done: StoppingCriterion,
        progress: bool = True,
    ):
        sampler = MSRSampler()
        super().__init__(
            utility=utility,
            sampler=sampler,
            is_done=is_done,
            progress=progress,
        )

    @staticmethod
    def coefficient(n: int, k: int) -> float:
        return 1.0

    def fit(self, data: Dataset) -> Self:

        # initialize an intermediate result object for positive updates
        self._pos_result = ValuationResult.zeros(
            indices=data.indices,
            data_names=data.data_names,
            algorithm=self.algorithm_name,
        )

        # initialize an intermediate result object for negative updates
        self._neg_result = ValuationResult.zeros(
            indices=data.indices,
            data_names=data.data_names,
            algorithm=self.algorithm_name,
        )

        ensure_backend_has_generator_return()

        self.utility.training_data = data

        parallel = Parallel(return_as="generator_unordered")
        strategy = self.sampler.make_strategy(self.utility, self.coefficient)
        processor = delayed(strategy.process)

        with make_parallel_flag() as flag:
            delayed_evals = parallel(
                processor(batch=list(batch), is_interrupted=flag)
                for batch in self.sampler.generate_batches(data.indices)
            )
            for batch in Progress(delayed_evals, self.is_done, **self.tqdm_args):
                for evaluation in batch:
                    # update the intermediate result objects
                    if evaluation.kind == ValueUpdateKind.POSITVE:
                        self._pos_result.update(evaluation.idx, evaluation.update)
                    elif evaluation.kind == ValueUpdateKind.NEGATIVE:
                        self._neg_result.update(evaluation.idx, evaluation.update)
                    else:
                        raise ValueError("Invalid ValueUpdateKind: {evaluation.kind}")

                    # combine the intermediate results into the final result
                    self.result = _combine_results(
                        self._pos_result, self._neg_result, data=data
                    )

                    if self.is_done(self.result):
                        flag.set()
                        self.sampler.interrupt()
                        break

                if self.is_done(self.result):
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

        return self


def _combine_results(
    pos_result: ValuationResult, neg_result: ValuationResult, data: Dataset
) -> ValuationResult:
    """Combine the positive and negative running means into a final result.

    We cannot simply subtract the negative result from the positive result because
    this would lead to wrong variance estimates, misleading update counts and even
    wrong values if no further precaution is taken.

    """
    # set counts to the minimum of the two; This enables us to ensure via stopping
    # criteria that both running means have a minimal number of updates
    counts = np.minimum(pos_result.counts, neg_result.counts)

    values = pos_result.values - neg_result.values
    # follow the convention to set the value to zero if one of the two counts was zero
    # TODO: Should this be nan instead?
    values[counts == 0] = 0.0

    variances = pos_result.variances + neg_result.variances
    # set the variance to infinity if one of the two counts was zero
    variances[counts == 0] = np.inf

    result = ValuationResult(
        values=values,
        variances=variances,
        counts=counts,
        indices=data.indices,
        data_names=data.data_names,
        algorithm=pos_result.algorithm,
    )

    return result

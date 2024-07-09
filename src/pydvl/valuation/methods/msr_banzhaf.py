r"""
This module implements the MSR-Banzhaf valuation method, as described in
(Wang et. al.)<sup><a href="#wang_data_2023">1</a></sup>.

## References

[^1]: <a name="wang_data_2023"></a>Wang, J.T. and Jia, R., 2023.
    [Data Banzhaf: A Robust Data Valuation Framework for Machine Learning](
    https://proceedings.mlr.press/v206/wang23e.html).
    In: Proceedings of The 26th International Conference on Artificial Intelligence and
    Statistics, pp. 6388-6421.

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
from pydvl.valuation.utility.base import UtilityBase
from pydvl.valuation.utils import (
    ensure_backend_has_generator_return,
    make_parallel_flag,
)

__all__ = ["MSRBanzhafValuation"]


class MSRBanzhafValuation(SemivalueValuation):
    """Class to compute Maximum Sample Re-use (MSR) Banzhaf values.

    See [Data Valuation][data-valuation] for an overview.

    The MSR Banzhaf valuation approximates the Banzhaf valuation and is much more
    efficient than traditional Montecarlo approaches.

    Args:
        utility: Utility object with model, data and scoring function.
        sampler: Sampling scheme to use. Currently, only one MSRSampler is implemented.
            In the future, weighted MSRSamplers will be supported.
        is_done: Stopping criterion to use.
        progress: Whether to show a progress bar.

    """

    algorithm_name = "MSR-Banzhaf"

    def __init__(
        self,
        utility: UtilityBase,
        sampler: MSRSampler,
        is_done: StoppingCriterion,
        progress: bool = True,
    ):
        super().__init__(
            utility=utility,
            sampler=sampler,
            is_done=is_done,
            progress=progress,
        )

    def coefficient(self, n: int, k: int) -> float:
        return 1.0

    def fit(self, data: Dataset) -> Self:
        """Calculate the MSR Banzhaf valuation on a dataset.

        This method has to be called before calling `values()`.

        Calculating the Banzhaf valuation is a computationally expensive task that
        can be parallelized. To do so, call the `fit()` method inside a
        `joblib.parallel_config` context manager as follows:

        ```python
        from joblib import parallel_config

        with parallel_config(n_jobs=4):
            valuation.fit(data)
        ```

        """
        pos_result = ValuationResult.zeros(
            indices=data.indices,
            data_names=data.data_names,
            algorithm=self.algorithm_name,
        )

        neg_result = ValuationResult.zeros(
            indices=data.indices,
            data_names=data.data_names,
            algorithm=self.algorithm_name,
        )

        self.result = ValuationResult.zeros(
            indices=data.indices,
            data_names=data.data_names,
            algorithm=self.algorithm_name,
        )

        ensure_backend_has_generator_return()

        self.utility.training_data = data

        strategy = self.sampler.make_strategy(self.utility, self.coefficient)
        processor = delayed(strategy.process)

        with Parallel(return_as="generator_unordered") as parallel:
            with make_parallel_flag() as flag:
                delayed_evals = parallel(
                    processor(batch=list(batch), is_interrupted=flag)
                    for batch in self.sampler.generate_batches(data.indices)
                )
                for batch in Progress(delayed_evals, self.is_done, **self.tqdm_args):
                    for evaluation in batch:
                        if evaluation.is_positive:
                            pos_result.update(evaluation.idx, evaluation.update)
                        else:
                            neg_result.update(evaluation.idx, evaluation.update)

                        self.result = self._combine_results(
                            pos_result, neg_result, data=data
                        )

                        if self.is_done(self.result):
                            flag.set()
                            self.sampler.interrupt()
                            break

                    if self.is_done(self.result):
                        break

        return self

    @staticmethod
    def _combine_results(
        pos_result: ValuationResult, neg_result: ValuationResult, data: Dataset
    ) -> ValuationResult:
        """Combine the positive and negative running means into a final result.

        Since MSR-Banzhaf values are not a mean over marginals, both the variances of
        the marginals and the update counts are ill-defined. We use the following
        conventions:

        1. The counts are defined as the minimum of the two counts. This definition
        enables us to ensure a minimal number of updates for both running means via
        stopping criteria and correctly detects that no actual update has taken place if
        one of the counts is zero.
        2. We reverse engineer the variances such that they yield correct standard
        errors given our convention for the counts and the normal calculation of
        standard errors in the valuation result.

        Note that we cannot use the normal addition or subtraction defined by the
        ValuationResult because it is weighted with counts. If we were to simply
        subtract the negative result from the positive we would get wrong variance
        estimates, misleading update counts and even wrong values if no further
        precaution is taken.

        TODO: Verify that the two running means are statistically independent (which is
 assumed in the aggregation of variances).

        Args:
            pos_result: The result of the positive updates.
            neg_result: The result of the negative updates.
            data: The dataset used for the valuation. Used for indices and names.

        Returns:
            The combined valuation result.

        """
        # define counts as minimum of the two counts (see docstring)
        counts = np.minimum(pos_result.counts, neg_result.counts)

        values = pos_result.values - neg_result.values
        values[counts == 0] = np.nan

        # define variances that yield correct standard errors (see docstring)
        pos_var = pos_result.variances / np.clip(pos_result.counts, 1, np.inf)
        neg_var = neg_result.variances / np.clip(neg_result.counts, 1, np.inf)
        variances = np.where(counts != 0, (pos_var + neg_var) * counts, np.inf)

        result = ValuationResult(
            values=values,
            variances=variances,
            counts=counts,
            indices=data.indices,
            data_names=data.data_names,
            algorithm=pos_result.algorithm,
        )

        return result

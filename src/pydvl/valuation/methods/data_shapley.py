r"""
This module implements the Data-Shapley valuation method.

The Data-Shapley method computes Shapley values by sampling sets of training points and
computing the marginal contribution of each element in these to the utility function.
This utility is the performance of a model trained on the sample, computed over a fixed
test set, and hence it is typically costly to compute.

Computing values always follows the same pattern: construct a
[ModelUtility][pydvl.valuation.utility.model.ModelUtility], a
[sampler][pydvl.valuation.samplers], and a
[stopping criterion][pydvl.valuation.stopping].

!!! Example "General usage pattern"
    ```python
    from pydvl.valuation import (
        DataShapleyValuation,
        ModelUtility,
        SupervisedScorer,
        PermutationSampler,
        RankCorrelation
    )

    model = SomeSKLearnModel()
    scorer = SupervisedScorer("accuracy", test_data, default=0)
    utility = ModelUtility(model, scorer, ...)
    sampler = PermutationSampler()
    stopping = RankCorrelation(rtol=0.01)
    valuation = DataShapleyValuation(utility, sampler, is_done=stopping)
    with parallel_config(n_jobs=16):
        valuation.fit(training_data)
    result = valuation.values()
    ```

## Choosing samplers

Different choices of sampler yield different qualities of approximation.

The most basic one is
[DeterministicUniformSampler][pydvl.valuation.samplers.DeterministicUniformSampler],
which iterates over all possible subsets of the training set. This is the most accurate,
but also the most computationally expensive method (with complexity $O(2^n)$), so it is
never used in practice.

The most common one is [PermutationSampler][pydvl.valuation.samplers.PermutationSampler],
which samples random permutations of the training set. Despite the apparent greater
complexity of $O(n!)$, the method is much faster to converge in practice, especially
when using [truncation policies][pydvl.valuation.samplers.truncation] to early-stop the
processing of each permutation.

Other samplers introduce altogether different ways of computing Shapley values, like
the [Owen samplers][pydvl.valuation.samplers.owen] or the
[Maximum-Sample-Reuse sampler][pydvl.valuation.samplers.MSRSampler], but the usage
pattern remains the same.

## Caveats

1. As mentioned, computing Shapley values can be computationally expensive, especially
   for large datasets. Some samplers yield better convergence, but not in all cases.
   Proper choice of a stopping criterion is crucial to obtain useful results, while
   avoiding unnecessary computation.
2. While it is possible to mix-and-match different components of the valuation method,
   it is not always advisable, and it can sometimes be incorrect. For example, using a
   deterministic sampler with a count-based stopping criterion is likely to yield poor
   results. More importantly, not all samplers, nor sampler configurations, are
   compatible with Shapley value computation. For instance using
   [NoIndexIteration][pydvl.valuation.samplers.NoIndexIteration] with a
   [PowerSetSampler][pydvl.valuation.samplers.PowerSetSampler] will not work since the
   evaluation strategy expects samples consisting of an index and a subset of its
   complement in the whole index set.
"""

import math

from pydvl.valuation.methods.semivalue import SemivalueValuation

__all__ = ["DataShapleyValuation"]


class DataShapleyValuation(SemivalueValuation):
    """Computes Shapley values."""

    algorithm_name = "Data-Shapley"

    def coefficient(self, n: int, k: int, weight: float) -> float:
        return weight / math.comb(n - 1, k) / n

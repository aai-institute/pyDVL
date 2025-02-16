r"""
This module implements the Shapley valuation method.

The (Data-)Shapley method, introduced in Ghorbani and Zou (2019)[^1] is a method to
compute data values by sampling sets of training points and averaging the change in
performance of a model by adding individual points to these sets. The average is done
using the Shapley coefficients, which correspond to the sampling probabilities of the
subsets used:

$$
v(i) = \frac{1}{n} \sum_{S \subset D_{-i}} w(n, |S|) [U(S_{+i}) - U(S)],
$$

where the coefficient $w(n, k)$ is defined as the inverse probability of sampling a set
of size $k$ from a set of size $n-1$ in the complement of $\{i\}$

$$
w(n, k) = \binom{n-1}{k}^{-1}.
$$

An alternative formulation, which has better variance properties, uses permutations. The
algorithm **Data-Shapley** described in Ghorbani and Zou (2019)[^1] uses this sampling
technique, together with a heuristic truncation policy to stop the computation early.
This is implemented in PyDVL via
[PermutationSampler][pydvl.valuation.samplers.PermutationSampler]

## Computing Shapley values

Computing values in PyDVL always follows the same pattern: construct a
[ModelUtility][pydvl.valuation.utility.modelutility.ModelUtility], a
[sampler][pydvl.valuation.samplers], and a
[stopping criterion][pydvl.valuation.stopping].

??? Example "General usage pattern"
    ```python
    from pydvl.valuation import (
        ShapleyValuation,
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
    valuation = ShapleyValuation(utility, sampler, is_done=stopping)
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

??? Example "Truncated Monte Carlo Data-Shapley"
    To compute Shapley values as described in Ghobani and Zou (2019)[^1], use this
    configuration:

    ```python
    truncation = RelativeTruncation(rtol=0.05)
    sampler = PermutationSampler(truncation=truncation, seed=seed)
    stopping = HistoryDeviation(n_steps=100, rtol=0.05)
    valuation = ShapleyValuation(utility, sampler, stopping, skip_converged, progress)
    ```

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

## References

[^1]: <a name="ghorbani_data_2019"></a>Ghorbani, A., & Zou, J. Y. (2019). [Data Shapley:
      Equitable Valuation
      of Data for Machine Learning](https://proceedings.mlr.press/v97/ghorbani19c.html).
      In Proceedings of the 36th International Conference on Machine Learning, PMLR pp.
      2242--2251.
"""

import numpy as np

from pydvl.utils import logcomb
from pydvl.valuation.methods.semivalue import SemivalueValuation

__all__ = ["ShapleyValuation"]


class ShapleyValuation(SemivalueValuation):
    """Computes Shapley values."""

    algorithm_name = "Shapley"

    def log_coefficient(self, n: int, k: int) -> float:
        return float(-np.log(n) - logcomb(n - 1, k))

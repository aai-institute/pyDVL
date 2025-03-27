r"""
This module implements the $\delta$-Shapley valuation method, introduced by Watson et al.
(2023)[^1].

!!! warning "Experimental"
    The $\delta$-Shapley value is an experimental feature. It has not been tested enough
    and is known to contain bugs. PRs welcome!

$\delta$-Shapley is a variant of the Shapley value that uses a stratified sampling
approach to skip computation of the marginal utilities for set sizes outside a small
range.

!!! info
    See [the documentation][delta-shapley-intro] or Watson et al. (2023)[^1] for a
    more detailed introduction to the method.

## References

[^1]: Watson, Lauren, Zeno Kujawa, Rayna Andreeva, Hao-Tsung Yang, Tariq Elahi, and Rik
      Sarkar. [Accelerated Shapley Value Approximation for Data
      Evaluation](https://doi.org/10.48550/arXiv.2311.05346). arXiv, 9 November 2023.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pydvl.valuation.methods.semivalue import SemivalueValuation
from pydvl.valuation.samplers import (
    StratifiedPermutationSampler,
    StratifiedSampler,
)
from pydvl.valuation.stopping import StoppingCriterion
from pydvl.valuation.types import SemivalueCoefficient
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["DeltaShapleyValuation"]


class DeltaShapleyValuation(SemivalueValuation):
    r"""Computes $\delta$-Shapley values.

    Args:
        utility: Object to compute utilities.
        sampler: The sampling scheme to use. Must be a stratified sampler.
        is_done: Stopping criterion to use.
        skip_converged: Whether to skip converged indices, as determined by the
            stopping criterion's `converged` array.
        show_warnings: Whether to show warnings.
        progress: Whether to show a progress bar. If a dictionary, it is passed to
            `tqdm` as keyword arguments, and the progress bar is displayed.
    """

    algorithm_name = "Delta-Shapley"
    sampler: StratifiedSampler | StratifiedPermutationSampler

    def __init__(
        self,
        utility: UtilityBase,
        sampler: StratifiedSampler | StratifiedPermutationSampler,
        is_done: StoppingCriterion,
        skip_converged: bool = False,
        show_warnings: bool = True,
        progress: dict[str, Any] | bool = False,
    ):
        super().__init__(
            utility, sampler, is_done, skip_converged, show_warnings, progress
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__}-"
            f"{self.utility.__class__.__name__}-"
            f"{self.sampler.__class__.__name__}-"
            f"{self.sampler.sample_sizes_strategy.__class__.__name__}-"
            f"{self.is_done}"
        )

    @property
    def log_coefficient(self) -> SemivalueCoefficient | None:
        r"""Returns the log-coefficient of the $\delta$-Shapley valuation.

        !!! note "Normalization"
            This coefficient differs from the one used in the original paper by a
            normalization factor of $m=\sum_k m_k,$ where $m_k$ is the number of
            samples of size $k$. Since this is constant across indices, the ranking of
            values is unaffected.
        """

        def _log_coefficient(n: int, k: int) -> float:
            effective_n = self.sampler.complement_size(n)
            lb, ub = self.sampler.sample_sizes_strategy.effective_bounds(effective_n)
            # We don't always have m_k so we use p_k instead, and return a coefficient
            # that is off by a constant factor m
            p = self.sampler.sample_sizes_strategy.sample_sizes(effective_n, probs=True)
            if p[k] == 0:
                return -np.inf
            # FIXME: why don't we subtract self.sampler.log_weight(n, k) here?
            return float(-np.log(ub - lb + 1) - np.log(p[k]) ) #- self.sampler.log_weight(n, k))

        return _log_coefficient

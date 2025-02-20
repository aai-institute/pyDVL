r"""
This module implements Leave-One-Out (LOO) valuation.

This is the simplest example of marginal-contribution-based valuation method. It is
defined as:

$$
v_\text{LOO}(i) = U(N) - U(N_{-i}),
$$

where $U$ is the utility function, $N$ is the set of all indices, and $i$ is the index
of interest.

Strictly speaking, LOO can be seen as a [semivalue][pydvl.valuation.methods.semivalue]
where the coefficients are zero except for $k=|D|-1.$

## Changing LOO

[LOOValuation][pydvl.valuation.methods.LOOValuation] is preconfigured to stop once all
indices have been visited once. In particular, it uses a default
[LOOSampler][pydvl.valuation.samplers.LOOSampler] with a
[FiniteSequentialIndexIteration][pydvl.valuation.iteration.FiniteSequentialIndexIteration].
If you want to change this behaviour, the easiest way is to subclass and replace the
constructor.

"""

from __future__ import annotations

import numpy as np

from pydvl.valuation.methods.semivalue import SemivalueValuation
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers import FiniteSequentialIndexIteration, LOOSampler
from pydvl.valuation.stopping import MinUpdates
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["LOOValuation"]


class LOOValuation(SemivalueValuation):
    """
    Computes LOO values for a dataset.
    """

    algorithm_name = "Leave-One-Out"

    def __init__(self, utility: UtilityBase, progress: bool = False):
        self.result: ValuationResult | None = None
        super().__init__(
            utility,
            LOOSampler(batch_size=1, index_iteration=FiniteSequentialIndexIteration),
            # LOO is done when every index has been updated once
            MinUpdates(n_updates=1),
            progress=progress,
        )

    def log_coefficient(self, n: int, k: int) -> float:
        """The LOOSampler returns only complements of {idx}, so the weight is either
        1/n (the probability of a set of size n-1) or 0 if k != n-1. We cancel
        this out here so that the final coefficient is either 1 if k == n-1 or 0
        otherwise."""
        return float(-np.log(max(1, n))) if k == n - 1 else -np.inf

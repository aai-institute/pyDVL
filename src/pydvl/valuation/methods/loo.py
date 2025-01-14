"""
This module implements Leave-One-Out (LOO) valuation.

This is the simplest example of marginal-contribution-based valuation method. It is
defined as:

$$
v_\text{LOO}(i) = U(I) - U(I \\setminus \\{i\\}),
$$

where $U$ is the utility function, $I$ is the set of all data points, and $i$ is the
data point of interest.

Strictly speaking, LOO can be seen as a [semivalue][pydvl.valuation.semivalue] where the
coefficients are zero except for $k=|D|-1$,
"""

from __future__ import annotations

from pydvl.valuation.methods.semivalue import SemivalueValuation
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers import LOOSampler
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
            LOOSampler(),
            # LOO is done when every index has been updated once
            MinUpdates(n_updates=1),
            progress=progress,
        )

    def coefficient(self, n: int, k: int) -> float:
        """
        This is never actually used to filter out sets, because the LOOSampler returns
        only complements of {idx}, but it is required by the abstract class.
        """
        return 1 if k == n - 1 else 0

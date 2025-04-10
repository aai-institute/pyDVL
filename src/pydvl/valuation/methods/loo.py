r"""
This module implements Leave-One-Out (LOO) valuation.

It is defined as:

$$
v_\text{loo}(i) = u(N) - u(N_{-i}),
$$

where $u$ is the utility function, $N$ is the set of all indices, and $i$ is the index
of interest.

## Changing LOO

[LOOValuation][pydvl.valuation.methods.LOOValuation] is preconfigured to stop once all
indices have been visited once. In particular, it uses a default
[LOOSampler][pydvl.valuation.samplers.LOOSampler] with a
[FiniteSequentialIndexIteration][pydvl.valuation.samplers.powerset.FiniteSequentialIndexIteration].
If you want to change this behaviour, the easiest way is to subclass and replace the
constructor.

"""

from __future__ import annotations

from pydvl.valuation.methods.semivalue import SemivalueValuation
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers import FiniteSequentialIndexIteration, LOOSampler
from pydvl.valuation.stopping import MinUpdates
from pydvl.valuation.types import SemivalueCoefficient
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["LOOValuation"]


class LOOValuation(SemivalueValuation):
    """
    Computes LOO values for a dataset.
    """

    algorithm_name = "Leave-One-Out"

    def __init__(self, utility: UtilityBase, progress: bool = False):
        self._result: ValuationResult | None = None
        super().__init__(
            utility,
            LOOSampler(batch_size=1, index_iteration=FiniteSequentialIndexIteration),
            # LOO is done when every index has been updated once
            MinUpdates(n_updates=1),
            progress=progress,
        )

    @property
    def log_coefficient(self) -> SemivalueCoefficient | None:
        """Disable importance sampling for this method since we have a fixed sampler
        that already provides the correct weights for the Monte Carlo approximation."""
        return None

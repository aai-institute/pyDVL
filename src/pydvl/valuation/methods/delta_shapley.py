from __future__ import annotations

import math

from pydvl.utils.types import Seed
from pydvl.valuation.methods.semivalue import SemivalueValuation
from pydvl.valuation.samplers import TruncatedUniformStratifiedSampler
from pydvl.valuation.stopping import StoppingCriterion

__all__ = ["DeltaShapleyValuation"]

from pydvl.valuation.utility.base import UtilityBase


class DeltaShapleyValuation(SemivalueValuation):
    r"""Computes $\delta$-Shapley values.

    $\delta$-Shapley does not accept custom samplers. Instead it uses a truncated
    hierarchical powerset sampler with a lower and upper bound on the size of the sets
    to sample from.

    TODO See ...
    """

    algorithm_name = "Delta-Shapley"

    def __init__(
        self,
        utility: UtilityBase,
        is_done: StoppingCriterion,
        lower_bound: int,
        upper_bound: int,
        seed: Seed | None = None,
        progress: bool = False,
    ):
        sampler = TruncatedUniformStratifiedSampler(
            lower_bound=lower_bound, upper_bound=upper_bound, seed=seed
        )
        super().__init__(utility, sampler, is_done, progress=progress)
        raise NotImplementedError(
            "Delta-Shapley has not been properly implemented nor tested yet."
        )

    def coefficient(self, n: int, k: int, weight: float) -> float:
        return weight / math.comb(n - 1, k)

"""
This module implements the Weighted-Banzhaf valuation method, as described in
(Wang and Jia, 2022)<sup><a href="#li_robust_2023">1</a></sup>.

## References

[^1]: <a name="li_robust_2023"></a>Li, Weida, and Yaoliang Yu. [Robust Data Valuation
      with Weighted Banzhaf Values](https://openreview.net/forum?id=u359tNBpxF). In
      Proceedings of the Thirty-Seventh Conference on Neural Information Processing
      Systems. New Orleans, Louisiana, USA, 2023.
"""

from pydvl.valuation.methods.semivalue import SemivalueValuation
from pydvl.valuation.samplers import IndexSampler
from pydvl.valuation.stopping import StoppingCriterion
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["WeightedBanzhafValuation"]


class WeightedBanzhafValuation(SemivalueValuation):
    algorithm_name = "Weighted-Banzhaf"

    def __init__(
        self, utility: UtilityBase, sampler: IndexSampler, is_done: StoppingCriterion
    ):
        super().__init__(utility, sampler, is_done)
        raise NotImplementedError()

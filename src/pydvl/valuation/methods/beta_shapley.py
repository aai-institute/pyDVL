import scipy as sp

from pydvl.valuation.methods.semivalue import SemivalueValuation
from pydvl.valuation.samplers.base import IndexSampler
from pydvl.valuation.stopping import StoppingCriterion

__all__ = ["BetaShapleyValuation"]

from pydvl.valuation.utility.base import UtilityBase


class BetaShapleyValuation(SemivalueValuation):
    """Computes Beta-Shapley values."""

    algorithm_name = "Beta-Shapley"

    def __init__(
        self,
        utility: UtilityBase,
        sampler: IndexSampler,
        is_done: StoppingCriterion,
        alpha: float,
        beta: float,
        progress: bool = False,
    ):
        super().__init__(utility, sampler, is_done, progress=progress)

        self.alpha = alpha
        self.beta = beta
        self.const = sp.special.beta(alpha, beta)
        self.log_const = sp.special.betaln(alpha, beta)

    def log_coefficient(self, n: int, k: int) -> float:
        j = k + 1
        return float(
            sp.special.betaln(j + self.beta - 1, n - j + self.alpha) - self.log_const
        )

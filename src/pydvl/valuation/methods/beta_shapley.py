import scipy as sp

from pydvl.valuation.methods.semivalue import SemivalueValuation
from pydvl.valuation.samplers.base import IndexSampler
from pydvl.valuation.stopping import StoppingCriterion

__all__ = ["BetaShapleyValuation"]

from pydvl.valuation.utility.base import UtilityBase


class BetaShapleyValuation(SemivalueValuation):
    """Computes Beta-Shapley values.

    Args:
        utility: Object to compute utilities.
        sampler: Sampling scheme to use.
        is_done: Stopping criterion to use.
        skip_converged: Whether to skip converged indices. Convergence is determined
            by the stopping criterion's `converged` array.
        alpha: The alpha parameter of the Beta distribution.
        beta: The beta parameter of the Beta distribution.
        progress: Whether to show a progress bar. If a dictionary, it is passed to
            `tqdm` as keyword arguments, and the progress bar is displayed.

    """

    algorithm_name = "Beta-Shapley"

    def __init__(
        self,
        utility: UtilityBase,
        sampler: IndexSampler,
        is_done: StoppingCriterion,
        alpha: float,
        beta: float,
        progress: bool = False,
        skip_converged: bool = False,
    ):
        super().__init__(
            utility, sampler, is_done, skip_converged=skip_converged, progress=progress
        )

        self.alpha = alpha
        self.beta = beta
        self.const = sp.special.beta(alpha, beta)
        self.log_const = sp.special.betaln(alpha, beta)

    def log_coefficient(self, n: int, k: int) -> float:
        j = k + 1
        return float(
            sp.special.betaln(j + self.beta - 1, n - j + self.alpha) - self.log_const
        )

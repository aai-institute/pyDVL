"""
## References

[^1]: <a name="ghorbani_data_2019"></a>Ghorbani, A., Zou, J., 2019.
    [Data Shapley: Equitable Valuation of Data for Machine Learning](https://proceedings.mlr.press/v97/ghorbani19c.html).
    In: Proceedings of the 36th International Conference on Machine Learning, PMLR, pp. 2242–2251.

"""

import abc
import logging
from typing import Optional

import numpy as np

from pydvl.utils import Utility, running_moments

__all__ = [
    "TruncationPolicy",
    "NoTruncation",
    "FixedTruncation",
    "BootstrapTruncation",
    "RelativeTruncation",
]


logger = logging.getLogger(__name__)


class TruncationPolicy(abc.ABC):
    """A policy for deciding whether to stop computing marginals in a
    permutation.

    Statistics are kept on the number of calls and truncations as `n_calls` and
    `n_truncations` respectively.

    Attributes:
        n_calls: Number of calls to the policy.
        n_truncations: Number of truncations made by the policy.

    !!! Todo
        Because the policy objects are copied to the workers, the statistics
        are not accessible from the coordinating process. We need to add methods
        for this.
    """

    def __init__(self) -> None:
        self.n_calls: int = 0
        self.n_truncations: int = 0

    @abc.abstractmethod
    def _check(self, idx: int, score: float) -> bool:
        """Implement the policy."""
        ...

    @abc.abstractmethod
    def reset(self, u: Optional[Utility] = None):
        """Reset the policy to a state ready for a new permutation."""
        ...

    def __call__(self, idx: int, score: float) -> bool:
        """Check whether the computation should be interrupted.

        Args:
            idx: Position in the permutation currently being computed.
            score: Last utility computed.

        Returns:
            `True` if the computation should be interrupted.
        """
        ret = self._check(idx, score)
        self.n_calls += 1
        self.n_truncations += 1 if ret else 0
        return ret


class NoTruncation(TruncationPolicy):
    """A policy which never interrupts the computation."""

    def _check(self, idx: int, score: float) -> bool:
        return False

    def reset(self, u: Optional[Utility] = None):
        pass


class FixedTruncation(TruncationPolicy):
    """Break a permutation after computing a fixed number of marginals.

    The experiments in Appendix B of (Ghorbani and Zou, 2019)<sup><a href="#ghorbani_data_2019">1</a></sup>
    show that when the training set size is large enough, one can simply truncate the iteration
    over permutations after a fixed number of steps. This happens because beyond
    a certain number of samples in a training set, the model becomes insensitive
    to new ones. Alas, this strongly depends on the data distribution and the
    model and there is no automatic way of estimating this number.

    Args:
        u: Utility object with model, data, and scoring function
        fraction: Fraction of marginals in a permutation to compute before
            stopping (e.g. 0.5 to compute half of the marginals).
    """

    def __init__(self, u: Utility, fraction: float):
        super().__init__()
        if fraction <= 0 or fraction > 1:
            raise ValueError("fraction must be in (0, 1]")
        self.max_marginals = len(u.data) * fraction
        self.count = 0

    def _check(self, idx: int, score: float) -> bool:
        self.count += 1
        return self.count >= self.max_marginals

    def reset(self, u: Optional[Utility] = None):
        self.count = 0


class RelativeTruncation(TruncationPolicy):
    """Break a permutation if the marginal utility is too low.

    This is called "performance tolerance" in (Ghorbani and Zou, 2019)<sup><a href="#ghorbani_data_2019">1</a></sup>.

    Args:
        u: Utility object with model, data, and scoring function
        rtol: Relative tolerance. The permutation is broken if the
            last computed utility is less than `total_utility * rtol`.
    """

    def __init__(self, u: Utility, rtol: float):
        super().__init__()
        self.rtol = rtol
        logger.info("Computing total utility for permutation truncation.")
        self.total_utility = self.reset(u)
        self._u = u

    def _check(self, idx: int, score: float) -> bool:
        # Explicit cast for the benefit of mypy 🤷
        return bool(np.allclose(score, self.total_utility, rtol=self.rtol))

    def reset(self, u: Optional[Utility] = None):
        if u is None:
            u = self._u

        self.total_utility = u(u.data.indices)


class BootstrapTruncation(TruncationPolicy):
    """Break a permutation if the last computed utility is close to the total
    utility, measured as a multiple of the standard deviation of the utilities.

    Args:
        u: Utility object with model, data, and scoring function
        n_samples: Number of bootstrap samples to use to compute the variance
            of the utilities.
        sigmas: Number of standard deviations to use as a threshold.
    """

    def __init__(self, u: Utility, n_samples: int, sigmas: float = 1):
        super().__init__()
        self.n_samples = n_samples
        logger.info("Computing total utility for permutation truncation.")
        self.total_utility = u(u.data.indices)
        self.count: int = 0
        self.variance: float = 0
        self.mean: float = 0
        self.sigmas: float = sigmas

    def _check(self, idx: int, score: float) -> bool:
        self.mean, self.variance = running_moments(
            self.mean, self.variance, self.count, score
        )
        self.count += 1
        logger.info(
            f"Bootstrap truncation: {self.count} samples, {self.variance:.2f} variance"
        )
        if self.count < self.n_samples:
            return False
        return abs(score - self.total_utility) < float(
            self.sigmas * np.sqrt(self.variance)
        )

    def reset(self, u: Optional[Utility] = None):
        self.count = 0
        self.variance = self.mean = 0

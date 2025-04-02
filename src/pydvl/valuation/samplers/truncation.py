"""
Truncation policies for the interruption of batched computations.

When estimating values with a
[PermutationSampler][pydvl.valuation.samplers.permutation.PermutationSampler], it is
possible to interrupt the marginal utility updates for one permutation based on
different heuristics. A naive approach is to stop after a fixed number of updates, using
[FixedTruncation][pydvl.valuation.samplers.truncation.FixedTruncation].

However, a more successful one is to stop if the utility of the current batch of samples
is close enough to the total utility of the dataset. This idea is implemented in
[RelativeTruncation][pydvl.valuation.samplers.truncation.RelativeTruncation], and was
introduced as **Truncated Montecarlo Shapley** (TMCS) in Ghobani and Zou (2019)[^1]. A
slight variation is to use the standard deviation of the utilities to set the tolerance,
which can be done with
[DeviationTruncation][pydvl.valuation.samplers.truncation.DeviationTruncation].

!!! Danger "Stopping too early"
    Truncation policies can lead to underestimation of values if the utility function
    has high variance. To avoid this, one can set a burn-in period before checking the
    utility, or use a policy that is less sensitive to variance.


## References

[^1]: <a name="ghorbani_data_2019"></a>Ghorbani, A., Zou, J., 2019. [Data Shapley:
    Equitable Valuation of Data for Machine
    Learning](https://proceedings.mlr.press/v97/ghorbani19c.html).
    In: Proceedings of the 36th International Conference on Machine Learning, PMLR, pp.
    2242–2251.

"""

import logging
from abc import ABC, abstractmethod

import numpy as np

from pydvl.utils.numeric import running_moments
from pydvl.valuation.types import IndexT, Sample
from pydvl.valuation.utility.base import UtilityBase

__all__ = [
    "TruncationPolicy",
    "NoTruncation",
    "FixedTruncation",
    "RelativeTruncation",
    "DeviationTruncation",
]


logger = logging.getLogger(__name__)


class TruncationPolicy(ABC):
    """A policy for deciding whether to stop computation of a batch of samples

    Statistics are kept on the total number of calls and truncations as `n_calls` and
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

    @abstractmethod
    def _check(self, idx: IndexT, score: float, batch_size: int) -> bool:
        """Implement the policy."""
        ...

    @abstractmethod
    def reset(self, utility: UtilityBase):
        """(Re)set the policy to a state ready for a new permutation."""
        ...

    def __call__(self, idx: IndexT, score: float, batch_size: int) -> bool:
        """Check whether the computation should be interrupted.

        Args:
            idx: Position in the batch currently being computed.
            score: Last utility computed.
            batch_size: Size of the batch being computed.

        Returns:
            `True` if the computation should be interrupted.
        """

        ret = self._check(idx, score, batch_size)
        self.n_calls += 1
        self.n_truncations += 1 if ret else 0
        return ret


class NoTruncation(TruncationPolicy):
    """A policy which never interrupts the computation."""

    def _check(self, idx: IndexT, score: float, batch_size: int) -> bool:
        return False

    def reset(self, _: UtilityBase):
        pass


class FixedTruncation(TruncationPolicy):
    """Break a computation after a fixed number of updates.

    The experiments in Appendix B of (Ghorbani and Zou, 2019)<sup><a
    href="#ghorbani_data_2019">1</a></sup> show that when the training set size is large
    enough, one can simply truncate the iteration over permutations after a fixed number
    of steps. This happens because beyond a certain number of samples in a training set,
    the model becomes insensitive to new ones. Alas, this strongly depends on the data
    distribution and the model and there is no automatic way of estimating this number.

    Args:
        fraction: Fraction of updates in a batch to compute before stopping (e.g.
            0.5 to compute half of the marginals in a permutation).
    """

    def __init__(self, fraction: float):
        super().__init__()
        if fraction <= 0 or fraction > 1:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = fraction
        self.count = 0  # within-permutation count

    def _check(self, idx: IndexT, score: float, batch_size: int) -> bool:
        self.count += 1
        return self.count >= self.fraction * batch_size

    def reset(self, _: UtilityBase):
        self.count = 0


class RelativeTruncation(TruncationPolicy):
    """Break a computation if the utility is close enough to the total utility.

    This is called "performance tolerance" in (Ghorbani and Zou, 2019)<sup><a
    href="#ghorbani_data_2019">1</a></sup>.

    !!! Warning
        Initialization and `reset()` of this policy imply the computation of the total
        utility for the dataset, which can be expensive!

    Args:
        rtol: Relative tolerance. The permutation is broken if the
            last computed utility is within this tolerance of the total utility.
        burn_in_fraction: Fraction of samples within a permutation to wait until
            actually checking.
    """

    def __init__(self, rtol: float, burn_in_fraction: float = 0.0):
        super().__init__()
        assert 0 <= burn_in_fraction <= 1
        self.burn_in_fraction = burn_in_fraction
        self.rtol = rtol
        self.total_utility = 0.0
        self.count = 0  # within-permutation count
        self._is_setup = False

    def _check(self, idx: IndexT, score: float, batch_size: int) -> bool:
        self.count += 1
        if self.count < self.burn_in_fraction * batch_size:
            return False
        return np.allclose(score, self.total_utility, rtol=self.rtol)

    def __call__(self, idx: IndexT, score: float, batch_size: int) -> bool:
        if not self._is_setup:
            raise ValueError("RelativeTruncation not set up. Call reset() first.")
        return super().__call__(idx, score, batch_size)

    def reset(self, utility: UtilityBase):
        self.count = 0
        if self._is_setup:
            return
        logger.info("Computing total utility for RelativeTruncation.")
        assert utility.training_data is not None
        self.total_utility = utility(Sample(None, utility.training_data.indices))
        self._is_setup = True


class DeviationTruncation(TruncationPolicy):
    """Break a computation if the last computed utility is close to the total utility.

    This is essentially the same as
    [RelativeTruncation][pydvl.valuation.samplers.truncation.RelativeTruncation], but with
    the tolerance determined by a multiple of the standard deviation of the utilities.

    !!! Danger
        This policy can break early if the utility function has high variance. This can
        lead to gross underestimation of values. Use with caution.

    !!! Warning
        Initialization and `reset()` of this policy imply the computation of the total
        utility for the dataset, which can be expensive!

    Args:
        burn_in_fraction: Fraction of samples within a permutation to wait until
            actually checking.
        sigmas: Number of standard deviations to use as a threshold.
    """

    def __init__(self, sigmas: float, burn_in_fraction: float = 0.0):
        super().__init__()
        assert 0 <= burn_in_fraction <= 1

        self.burn_in_fraction = burn_in_fraction
        self.total_utility = 0.0
        self.count = 0  # within-permutation count
        self.variance = 0.0
        self.mean = 0.0
        self.sigmas = sigmas
        self._is_setup = False

    def _check(self, idx: IndexT, score: float, batch_size: int) -> bool:
        self.mean, self.variance = running_moments(
            self.mean, self.variance, self.count, score
        )
        self.count += 1
        logger.info(
            f"Bootstrap truncation: {self.count} samples, {self.variance:.2f} variance"
        )
        if self.count < self.burn_in_fraction * batch_size:
            return False
        return abs(score - self.total_utility) < float(
            self.sigmas * np.sqrt(self.variance)
        )

    def __call__(self, idx: IndexT, score: float, batch_size: int) -> bool:
        if not self._is_setup:
            raise ValueError("DeviationPolicy not set up. Call reset() first.")
        return super().__call__(idx, score, batch_size)

    def reset(self, utility: UtilityBase):
        self.count = 0
        self.variance = self.mean = 0.0
        if self._is_setup:
            return
        logger.info("Computing total utility for DeviationTruncation.")
        assert utility.training_data is not None
        self.total_utility = utility(Sample(None, utility.training_data.indices))
        self._is_setup = True

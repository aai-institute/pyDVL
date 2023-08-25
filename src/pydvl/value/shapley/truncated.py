import abc
import logging
from typing import cast

import numpy as np
from deprecate import deprecated

from pydvl.utils import ParallelConfig, Utility, running_moments
from pydvl.value import ValuationResult
from pydvl.value.stopping import StoppingCriterion

__all__ = [
    "TruncationPolicy",
    "NoTruncation",
    "FixedTruncation",
    "BootstrapTruncation",
    "RelativeTruncation",
    "truncated_montecarlo_shapley",
]


logger = logging.getLogger(__name__)


class TruncationPolicy(abc.ABC):
    """A policy for deciding whether to stop computing marginals in a
    permutation.

    Statistics are kept on the number of calls and truncations as :attr:`n_calls`
    and :attr:`n_truncations` respectively.

    .. todo::
       Because the policy objects are copied to the workers, the statistics
       are not accessible from the
       :class:`~pydvl.value.shapley.actor.ShapleyCoordinator`. We need to add
       methods for this.
    """

    def __init__(self):
        self.n_calls: int = 0
        self.n_truncations: int = 0

    @abc.abstractmethod
    def _check(self, idx: int, score: float) -> bool:
        """Implement the policy."""
        ...

    @abc.abstractmethod
    def reset(self):
        """Reset the policy to a state ready for a new permutation."""
        ...

    def __call__(self, idx: int, score: float) -> bool:
        """Check whether the computation should be interrupted.

        :param idx: Position in the permutation currently being computed.
        :param score: Last utility computed.
        :return: ``True`` if the computation should be interrupted.
        """
        ret = self._check(idx, score)
        self.n_calls += 1
        self.n_truncations += 1 if ret else 0
        return ret


class NoTruncation(TruncationPolicy):
    """A policy which never interrupts the computation."""

    def _check(self, idx: int, score: float) -> bool:
        return False

    def reset(self):
        pass


class FixedTruncation(TruncationPolicy):
    """Break a permutation after computing a fixed number of marginals.

    The experiments in Appendix B of :footcite:t:`ghorbani_data_2019` show
    that when the training set size is large enough, one can simply truncate the
    iteration over permutations after a fixed number of steps. This happens
    because beyond a certain number of samples in a training set, the model
    becomes insensitive to new ones. Alas, this strongly depends on the data
    distribution and the model and there is no automatic way of estimating this
    number.

    :param u: Utility object with model, data, and scoring function
    :param fraction: Fraction of marginals in a permutation to compute before
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

    def reset(self):
        self.count = 0


class RelativeTruncation(TruncationPolicy):
    """Break a permutation if the marginal utility is too low.

    This is called "performance tolerance" in :footcite:t:`ghorbani_data_2019`.

    :param u: Utility object with model, data, and scoring function
    :param rtol: Relative tolerance. The permutation is broken if the
        last computed utility is less than ``total_utility * rtol``.
    """

    def __init__(self, u: Utility, rtol: float):
        super().__init__()
        self.rtol = rtol
        logger.info("Computing total utility for permutation truncation.")
        self.total_utility = u(u.data.indices)

    def _check(self, idx: int, score: float) -> bool:
        return np.allclose(score, self.total_utility, rtol=self.rtol)

    def reset(self):
        pass


class BootstrapTruncation(TruncationPolicy):
    """Break a permutation if the last computed utility is close to the total
    utility, measured as a multiple of the standard deviation of the utilities.

    :param u: Utility object with model, data, and scoring function
    :param n_samples: Number of bootstrap samples to use to compute the variance
        of the utilities.
    :param sigmas: Number of standard deviations to use as a threshold.
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

    def reset(self):
        self.count = 0
        self.variance = self.mean = 0


@deprecated(
    target=True,
    deprecated_in="0.7.0",
    remove_in="0.8.0",
    args_mapping=dict(coordinator_update_period=None, worker_update_period=None),
)
def truncated_montecarlo_shapley(
    u: Utility,
    *,
    done: StoppingCriterion,
    truncation: TruncationPolicy,
    config: ParallelConfig = ParallelConfig(),
    n_jobs: int = 1,
    coordinator_update_period: int = 10,
    worker_update_period: int = 5,
) -> ValuationResult:
    """
    .. warning::
       This method is deprecated and only a wrapper for
       :func:`~pydvl.value.shapley.montecarlo.permutation_montecarlo_shapley`.

    :param u:
    :param done:
    :param truncation:
    :param config:
    :param n_jobs:
    :param coordinator_update_period:
    :param worker_update_period:
    :return:
    """
    from pydvl.value.shapley.montecarlo import permutation_montecarlo_shapley

    return cast(
        ValuationResult,
        permutation_montecarlo_shapley(
            u, done=done, truncation=truncation, config=config, n_jobs=n_jobs
        ),
    )

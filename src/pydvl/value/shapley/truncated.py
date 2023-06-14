import abc
import logging
from concurrent.futures import FIRST_COMPLETED, wait

import numpy as np
from deprecate import deprecated

from pydvl.utils import ParallelConfig, Utility, running_moments
from pydvl.utils.parallel.backend import effective_n_jobs, init_parallel_backend
from pydvl.utils.parallel.futures import init_executor
from pydvl.value import ValuationResult
from pydvl.value.stopping import MaxChecks, StoppingCriterion

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

    !!! Todo
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

        Args:
            idx: Position in the permutation currently being computed.
            score: Last utility computed.

        Returns:
            ``True`` if the computation should be interrupted.
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

    def reset(self):
        self.count = 0


class RelativeTruncation(TruncationPolicy):
    """Break a permutation if the marginal utility is too low.

    This is called "performance tolerance" in :footcite:t:`ghorbani_data_2019`.

    Args:
        u: Utility object with model, data, and scoring function
        rtol: Relative tolerance. The permutation is broken if the
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

    def reset(self):
        self.count = 0
        self.variance = self.mean = 0


def _permutation_montecarlo_one_step(
    u: Utility,
    truncation: TruncationPolicy,
    algorithm: str,
) -> ValuationResult:
    # Avoid circular imports
    from .montecarlo import _permutation_montecarlo_shapley

    result = _permutation_montecarlo_shapley(
        u,
        done=MaxChecks(1),
        truncation=truncation,
        algorithm_name=algorithm,
    )
    nans = np.isnan(result.values).sum()
    if nans > 0:
        logger.warning(
            f"{nans} NaN values in current permutation, ignoring. "
            "Consider setting a default value for the Scorer"
        )
        result = ValuationResult.empty(algorithm="truncated_montecarlo_shapley")
    return result


@deprecated(
    target=True,
    deprecated_in="0.6.1",
    remove_in="0.7.0",
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
    """Monte Carlo approximation to the Shapley value of data points.

    This implements the permutation-based method described in
    :footcite:t:`ghorbani_data_2019`. It is a Monte Carlo estimate of the sum
    over all possible permutations of the index set, with a double stopping
    criterion.

    !!! Todo
        Think of how to add Robin-Gelman or some other more principled stopping
        criterion.

    Instead of naively implementing the expectation, we sequentially add points
    to a dataset from a permutation and incrementally compute marginal utilities.
    We stop computing marginals for a given permutation based on a
    :class:`TruncationPolicy`. :footcite:t:`ghorbani_data_2019` mention two
    policies: one that stops after a certain fraction of marginals are computed,
    implemented in :class:`FixedTruncation`, and one that stops if the last
    computed utility ("score") is close to the total utility using the standard
    deviation of the utility as a measure of proximity, implemented in
    :class:`BootstrapTruncation`.

    We keep sampling permutations and updating all shapley values
    until the :class:`StoppingCriterion` returns ``True``.

    Args:
        u: Utility object with model, data, and scoring function
        done: Check on the results which decides when to stop sampling
            permutations.
        truncation: callable that decides whether to stop computing marginals
            for a given permutation.
        config: Object configuring parallel computation, with cluster address,
            number of cpus, etc.
        n_jobs: Number of permutation monte carlo jobs to run concurrently.
        coordinator_update_period: in seconds. How often to check the
            accumulated results from the workers for convergence.
        worker_update_period: interval in seconds between different updates to
            and from the coordinator

    Returns:
        Object with the data values.
    """
    algorithm = "truncated_montecarlo_shapley"

    parallel_backend = init_parallel_backend(config)
    u = parallel_backend.put(u)
    # This represents the number of jobs that are running
    n_jobs = effective_n_jobs(n_jobs, config)
    # This determines the total number of submitted jobs
    # including the ones that are running
    n_submitted_jobs = 2 * n_jobs

    accumulated_result = ValuationResult.zeros(algorithm=algorithm)

    with init_executor(max_workers=n_jobs, config=config) as executor:
        futures = set()
        # Initial batch of computations
        for _ in range(n_submitted_jobs):
            future = executor.submit(
                _permutation_montecarlo_one_step,
                u,
                truncation,
                algorithm,
            )
            futures.add(future)
        while futures:
            # Wait for the next futures to complete.
            completed_futures, futures = wait(
                futures, timeout=60, return_when=FIRST_COMPLETED
            )
            for future in completed_futures:
                accumulated_result += future.result()
                if done(accumulated_result):
                    break
            if done(accumulated_result):
                break
            # Submit more computations
            # The goal is to always have `n_jobs`
            # computations running
            for _ in range(n_submitted_jobs - len(futures)):
                future = executor.submit(
                    _permutation_montecarlo_one_step,
                    u,
                    truncation,
                    algorithm,
                )
                futures.add(future)
    return accumulated_result

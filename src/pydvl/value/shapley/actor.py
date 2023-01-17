"""
Methods and classes to distribute jobs computing Shapley values in a cluster.

You probably aren't interested in any of this unless you are developing new
methods for pyDVL that use parallelization.
"""

import logging
import operator
import warnings
from functools import reduce
from time import time
from typing import Optional, cast

import numpy as np

from ...utils.config import ParallelConfig
from ...utils.parallel.actor import Coordinator, RayActorWrapper, Worker
from ...utils.parallel.backend import RayParallelBackend, init_parallel_backend
from ...utils.status import Status
from ...utils.utility import Utility
from ...value.results import ValuationResult
from ..convergence import ConvergenceCheck

__all__ = ["get_shapley_coordinator", "get_shapley_worker"]


logger = logging.getLogger(__name__)


def get_shapley_coordinator(
    *args, config: ParallelConfig = ParallelConfig(), **kwargs
) -> "ShapleyCoordinator":
    if config.backend == "ray":
        parallel_backend = cast(RayParallelBackend, init_parallel_backend(config))
        remote_cls = parallel_backend.wrap(ShapleyCoordinator)
        handle = remote_cls.remote(*args, **kwargs)
        coordinator = cast(
            ShapleyCoordinator, RayActorWrapper(handle, parallel_backend)
        )
    elif config.backend == "sequential":
        coordinator = ShapleyCoordinator(*args, **kwargs)
    else:
        raise NotImplementedError(f"Unexpected parallel type {config.backend}")
    return coordinator


def get_shapley_worker(
    *args, config: ParallelConfig = ParallelConfig(), **kwargs
) -> "ShapleyWorker":
    if config.backend == "ray":
        parallel_backend = cast(RayParallelBackend, init_parallel_backend(config))
        remote_cls = parallel_backend.wrap(ShapleyWorker)
        handle = remote_cls.remote(*args, **kwargs)
        worker = cast(ShapleyWorker, RayActorWrapper(handle, parallel_backend))
    elif config.backend == "sequential":
        worker = ShapleyWorker(*args, **kwargs)
    else:
        raise NotImplementedError(f"Unexpected parallel type {config.backend}")
    return worker


class ShapleyCoordinator(Coordinator):
    """The coordinator has two main tasks: aggregating the results of the
    workers and terminating processes once a certain stopping criterion is
    satisfied.
    """

    def __init__(self, convergence_check: ConvergenceCheck):
        super().__init__()
        self.convergence_check = convergence_check

    def accumulate(self) -> Optional[ValuationResult]:
        """Accumulates all results received from the workers.

        :return: Values and standard errors in a
            :class:`~pydvl.value.results.ValuationResult`. If no worker has
            reported yet, returns ``None``.
        """
        if len(self.worker_results) == 0:
            return None

        # FIXME: very inefficient
        totals = reduce(operator.add, self.worker_results)
        # Avoid recomputing
        self.worker_results = [totals]
        return totals

    def check_done(self) -> bool:
        """Checks whether the accuracy of the calculation or the total number
        of iterations have crossed the set thresholds.

        If any of the thresholds have been reached, then calls to
        :meth:`~Coordinator.is_done` return `True`.

        :return: True if converged or reached max iterations.
        """
        if self.is_done():
            return True

        if len(self.worker_results) > 0:
            self._status = self.convergence_check(self.accumulate())

        return self.is_done()

    def status(self) -> Status:
        return self._status


class ShapleyWorker(Worker):
    """A worker.

    It should work.
    """

    def __init__(
        self,
        u: Utility,
        coordinator: ShapleyCoordinator,
        *,
        worker_id: int,
        permutation_breaker: Optional[PermutationBreaker] = None,
        update_period: int = 30,
    ):
        """A worker calculates Shapley values using the permutation definition
         and reports the results to the coordinator.

         To implement early stopping, workers can be signaled by the
         :class:`~pydvl.value.shapley.actor.ShapleyCoordinator` before they are
         done with their work package

        :param u: Utility object with model, data, and scoring function
        :param coordinator: worker results will be pushed to this coordinator
        :param worker_id: id used for reporting through maybe_progress
        :param permutation_breaker: Stopping criteria to apply within
            individual permutations.
        :param update_period: interval in seconds between different updates to
            and from the coordinator
        """
        super().__init__(
            coordinator=coordinator, update_period=update_period, worker_id=worker_id
        )
        self.u = u
        self.permutation_breaker = permutation_breaker

    def _compute_marginals(self) -> ValuationResult:
        # Import here to avoid errors with circular imports
        from .montecarlo import _permutation_montecarlo_shapley

        return _permutation_montecarlo_shapley(
            self.u,
            permutation_breaker=self.permutation_breaker,
            algorithm_name="truncated_montecarlo_shapley",
        )

    def run(self, *args, **kwargs):
        """Computes marginal utilities in a loop until signalled to stop.

        This calls :meth:`_compute_marginals` repeatedly calculating Shapley
        values on different permutations of the indices. After ``update_period``
        seconds have passed, it reports the results to the
        :class:`~pydvl.value.shapley.actor.ShapleyCoordinator`. Before starting
        the next iteration, it checks the coordinator's
        :meth:`~pydvl.utils.parallel.actor.Coordinator.is_done` flag,
        terminating if it's ``True``.
        """
        acc = None
        while not self.coordinator.is_done():
            start_time = time()
            while (time() - start_time) < self.update_period:
                results = self._compute_marginals()
                if np.any(np.isnan(results.values)):
                    warnings.warn(
                        "NaN values in current permutation, ignoring",
                        RuntimeWarning,
                    )
                    continue
                acc = results if acc is None else acc + results
            self.coordinator.add_results(acc)

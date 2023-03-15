"""
Methods and classes to distribute jobs computing Shapley values in a cluster.

You probably aren't interested in any of this unless you are developing new
methods for pyDVL that use parallelization.
"""

import logging
from time import time
from typing import cast

import numpy as np
from ray.util.queue import Empty, Queue

from pydvl.utils.config import ParallelConfig
from pydvl.utils.parallel import init_parallel_backend
from pydvl.utils.parallel.actor import Coordinator, QueueType, RayActorWrapper, Worker
from pydvl.utils.utility import Utility
from pydvl.value.result import ValuationResult
from pydvl.value.shapley.truncated import TruncationPolicy
from pydvl.value.stopping import MaxChecks, StoppingCriterion

__all__ = ["get_shapley_coordinator", "get_shapley_worker"]


logger = logging.getLogger(__name__)


def get_shapley_queue(
    maxsize: int, config: ParallelConfig = ParallelConfig()
) -> QueueType:
    if config.backend == "ray":
        queue = Queue(maxsize=maxsize)
    else:
        raise NotImplementedError(f"Unexpected parallel type {config.backend}")
    return queue


def get_shapley_coordinator(
    *args, config: ParallelConfig = ParallelConfig(), **kwargs
) -> "ShapleyCoordinator":
    if config.backend == "ray":
        coordinator = cast(
            ShapleyCoordinator,
            RayActorWrapper(ShapleyCoordinator, config, *args, **kwargs),
        )
    else:
        raise NotImplementedError(f"Unexpected parallel type {config.backend}")
    return coordinator


def get_shapley_worker(
    u: Utility, *args, config: ParallelConfig = ParallelConfig(), **kwargs
) -> "ShapleyWorker":
    parallel_backend = init_parallel_backend(config)
    u_id = parallel_backend.put(u)
    if config.backend == "ray":
        worker = cast(
            ShapleyWorker, RayActorWrapper(ShapleyWorker, config, u_id, *args, **kwargs)
        )
    else:
        raise NotImplementedError(f"Unexpected parallel type {config.backend}")
    return worker


class ShapleyCoordinator(Coordinator):
    """The coordinator has two main tasks: aggregating the results of the
    workers and shutting down the queue once a certain stopping criterion is
    satisfied.

    :param queue: Used by workers to report their results to the coordinator.
    :param done: Stopping criterion.
    :param update_period: Interval in seconds in-between convergence checks.
    """

    def __init__(
        self, queue: QueueType, done: StoppingCriterion, *, update_period: int = 10
    ):
        super().__init__(queue=queue)
        self.update_period = update_period
        self.result = ValuationResult.empty()
        self.results_done = done
        self.results_done.modify_result = True

    def check_convergence(self) -> bool:
        """Evaluates the convergence criterion on the aggregated results.

        If the convergence criterion is satisfied, calls to
        :meth:`~Coordinator.is_done` return ``True``.

        :return: ``True`` if converged and ``False`` otherwise.
        """
        if self.is_done():
            return True
        self._status = self.results_done(self.result)
        return self.is_done()

    def run(self, *args, **kwargs) -> ValuationResult:
        """Runs the coordinator."""
        start_time = time()
        while True:
            while (time() - start_time) < self.update_period:
                try:
                    worker_result: ValuationResult = self.queue.get(
                        block=True, timeout=30
                    )
                    self.result += worker_result
                except Empty:
                    break
            start_time = time()
            self._status = self.results_done(self.result)
            if self.check_convergence():
                break
        self.queue.shutdown()
        return self.result  # type: ignore


class ShapleyWorker(Worker):
    """A worker calculates Shapley values using the permutation definition,
    aggregates the results and puts them in the queue every
    `update_period` seconds until it is closed by the coordinator.

    To implement early stopping, workers can be signaled by the
    :class:`~pydvl.value.shapley.actor.ShapleyCoordinator` before they are
    done with their work package.


    :param u: Utility object with model, data, and scoring function
    :param queue: Used by workers to report their results to the coordinator.
    :param truncation: callable that decides whether to stop computing
        marginals for a given permutation.
    :param worker_id: id used for reporting through maybe_progress
    :param update_period: interval in seconds in-between updates to the queue.
    """

    algorithm: str = "truncated_montecarlo_shapley"

    def __init__(
        self,
        u: Utility,
        queue: QueueType,
        *,
        truncation: TruncationPolicy,
        worker_id: int,
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
        :param update_period: interval in seconds between different updates to
            and from the coordinator
        :param truncation: callable that decides whether to stop computing
            marginals for a given permutation.
        """
        super().__init__(queue=queue, update_period=update_period, worker_id=worker_id)
        self.u = u
        self.truncation = truncation

    def _compute_marginals(self) -> ValuationResult:
        # Avoid circular imports
        from .montecarlo import _permutation_montecarlo_shapley

        return _permutation_montecarlo_shapley(
            self.u,
            done=MaxChecks(1),
            truncation=self.truncation,
            algorithm_name=self.algorithm,
        )

    def run(self, *args, **kwargs):
        """Computes marginal utilities in a loop until signalled to stop.

        This calls :meth:`_compute_marginals` repeatedly calculating Shapley
        values on different permutations of the indices. After :attr:`update_period`
        seconds have passed, it puts the results in the queue.
        The loop is terminated if the queue is closed by the coordinator.
        """
        while True:
            acc = ValuationResult.empty()
            start_time = time()
            while (time() - start_time) < self.update_period:
                results = self._compute_marginals()
                nans = np.isnan(results.values).sum()
                if nans > 0:
                    logger.warning(
                        f"{nans} NaN values in current permutation, ignoring. "
                        "Consider setting a default value for the Scorer"
                    )
                    continue
                acc += results
            try:
                self.queue.put(acc, block=True, timeout=30)
            except Exception:
                break

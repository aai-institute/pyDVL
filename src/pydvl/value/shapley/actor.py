"""
Methods and classes to distribute jobs computing Shapley values in a cluster.

You probably aren't interested in any of this unless you are developing new
methods for pyDVL that use parallelization.
"""

import logging
import operator
from functools import reduce
from time import time
from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray

from ...utils.config import ParallelConfig
from ...utils.parallel.actor import Coordinator, RayActorWrapper, Worker
from ...utils.utility import Utility
from ...value.result import ValuationResult
from ..stopping import MaxUpdates, StoppingCriterion

__all__ = ["get_shapley_coordinator", "get_shapley_worker"]


logger = logging.getLogger(__name__)


def get_shapley_coordinator(
    *args, config: ParallelConfig = ParallelConfig(), **kwargs
) -> "ShapleyCoordinator":
    if config.backend == "ray":
        coordinator = cast(
            ShapleyCoordinator,
            RayActorWrapper(ShapleyCoordinator, config, *args, **kwargs),
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
        worker = cast(
            ShapleyWorker, RayActorWrapper(ShapleyWorker, config, *args, **kwargs)
        )
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

    def __init__(self, stop: StoppingCriterion):
        super().__init__()
        self.stop = stop
        self.stop.modify_result = True

    def accumulate(self) -> ValuationResult:
        """Accumulates all results received from the workers.

        :return: Values and standard errors in a
            :class:`~pydvl.value.result.ValuationResult`. If no worker has
            reported yet, returns ``None``.
        """
        if len(self.worker_results) == 0:
            return ValuationResult.empty()

        # FIXME: inefficient, possibly unstable
        totals: ValuationResult = reduce(operator.add, self.worker_results)
        # Avoid recomputing
        self.worker_results = [totals]
        return totals

    def check_convergence(self) -> bool:
        """Evaluates the convergence criterion on the accumulated results.

        If the convergence criterion is satisfied, calls to
        :meth:`~Coordinator.is_done` return ``True``.

        :return: ``True`` if converged and ``False`` otherwise.
        """
        if self.is_done():
            return True
        if len(self.worker_results) > 0:
            self._status = self.stop(self.accumulate())
        return self.is_done()


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
        update_period: int = 30,
        total_utility: Optional[float] = None,
        permutation_tolerance: Optional[float] = None,
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
        :param total_utility: total utility of the utility function. Pass to
            avoid recomputing in each worker.
        :param permutation_tolerance: tolerance for the permutation breaking
            algorithm ("performance tolerance" in :footcite:t:`ghorbani_data_2019`).
            Leave empty to set to ``total_utility / len(u.data) / 100``
        """
        super().__init__(
            coordinator=coordinator, update_period=update_period, worker_id=worker_id
        )
        self.u = u
        self.total_utility = total_utility or u(u.data.indices)
        self.permutation_tolerance = (
            permutation_tolerance or self.total_utility / len(self.u.data) / 100
        )

    def permutation_breaker(self, idx: int, marginals: NDArray[np.float_]) -> bool:
        return (
            float(np.abs(marginals[idx] - self.total_utility))
            < self.permutation_tolerance
        )

    def _compute_marginals(self) -> ValuationResult:
        # Import here to avoid errors with circular imports
        from .montecarlo import _permutation_montecarlo_shapley

        return _permutation_montecarlo_shapley(
            self.u,
            stop=MaxUpdates(1),
            permutation_breaker=self.permutation_breaker,
            algorithm_name="truncated_montecarlo_shapley",
        )

    def run(self, *args, **kwargs):
        """Computes marginal utilities in a loop until signalled to stop.

        This calls :meth:`_compute_marginals` repeatedly calculating Shapley
        values on different permutations of the indices. After :attr:`update_period`
        seconds have passed, it reports the results to the
        :class:`~pydvl.value.shapley.actor.ShapleyCoordinator`. Before starting
        the next iteration, it checks the coordinator's
        :meth:`~pydvl.utils.parallel.actor.Coordinator.is_done` flag,
        terminating if it's ``True``.
        """
        while True:
            acc = ValuationResult.empty()
            start_time = time()
            while (time() - start_time) < self.update_period:
                if self.coordinator.is_done():
                    return
                results = self._compute_marginals()
                if np.any(np.isnan(results.values)):
                    logger.warning("NaN values in current permutation, ignoring")
                    continue
                acc += results
            self.coordinator.add_results(acc)

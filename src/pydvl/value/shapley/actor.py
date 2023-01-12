"""
Methods and classes to distribute jobs computing Shapley values in a cluster.

You probably aren't interested in any of this unless you are developing new
methods for pyDVL that use parallelization.
"""

import logging
import warnings
from time import time
from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import numpy as np

from pydvl.utils import Utility, get_running_avg_variance, maybe_progress
from pydvl.utils.config import ParallelConfig
from pydvl.utils.parallel.actor import Coordinator, RayActorWrapper, Worker
from pydvl.utils.parallel.backend import RayParallelBackend, init_parallel_backend
from pydvl.value.results import ValuationStatus

if TYPE_CHECKING:
    from numpy.typing import NDArray


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

    :param value_tolerance: Terminate all workers if the ratio of median
        standard error to median of value has dropped below this value.
    :param n_iterations: Terminate if the current number of permutations
        has exceeded this threshold.
     :param progress: Whether to display progress bars for each job.
    """

    def __init__(
        self,
        value_tolerance: Optional[float] = None,
        n_iterations: Optional[int] = None,
        progress: Optional[bool] = True,
    ):
        super().__init__(progress=progress)
        if value_tolerance is None and n_iterations is None:
            raise ValueError(
                "Either value_tolerance or n_iterations must be set as a"
                "stopping criterion"
            )
        self.value_tolerance = value_tolerance
        self.n_iterations = n_iterations
        self._status = ValuationStatus.Pending

    def get_results(self) -> Tuple["NDArray", "NDArray"]:
        """Aggregates the results of the different workers

        :return: returns average and standard deviation of the value. If no
            worker has reported yet, returns two empty arrays.
        """
        values = []
        stds = []
        iterations = []
        self._total_iterations = 0

        if len(self.workers_results) == 0:
            return np.array([]), np.array([])

        for _, result in self.workers_results.items():
            values.append(result["values"])
            stds.append(result["std"])
            iterations.append(result["num_iter"])
            self._total_iterations += int(result["num_iter"])

        num_workers = len(values)
        if num_workers > 1:
            value = np.average(values, axis=0, weights=iterations)
            std = np.sqrt(
                np.average(
                    (np.asarray(values) - value) ** 2, axis=0, weights=iterations
                )
            ) / (num_workers - 1) ** (1 / 2)
        else:
            value = values[0]
            std = stds[0]
        return value, std

    def check_done(self) -> bool:
        """Checks whether the accuracy of the calculation or the total number
        of iterations have crossed the set thresholds.

        If any of the thresholds have been reached, then calls to
        :meth:`~Coordinator.is_done` return `True`.

        :return: True if converged or reached max iterations.
        """
        if self._is_done:
            return True

        if len(self.workers_results) == 0:
            logger.info("No worker has updated its status yet.")
            self._is_done = False
        else:
            value, std = self.get_results()
            std_to_val_ratio = np.median(std) / np.median(value)
            if (
                self.value_tolerance is not None
                and std_to_val_ratio < self.value_tolerance
            ):
                self._is_done = True
                logger.info("Converged")
                self._status = ValuationStatus.Converged
            elif (
                self.n_iterations is not None
                and self._total_iterations > self.n_iterations
            ):
                self._is_done = True
                logger.info(f"Max iterations ({self.n_iterations}) reached")
                self._status = ValuationStatus.MaxIterations
        return self._is_done

    def status(self) -> ValuationStatus:
        return self._status


class ShapleyWorker(Worker):
    """A worker.

    It should work.
    """

    def __init__(
        self,
        u: Utility,
        coordinator: ShapleyCoordinator,
        worker_id: int,
        *,
        update_period: int = 30,
        progress: bool = False,
    ):
        """A worker calculates Shapley values using the permutation definition
         and report the results to the coordinator.

        :param u: Utility object with model, data, and scoring function
        :param coordinator: worker results will be pushed to this coordinator
        :param worker_id: id used for reporting through maybe_progress
        :param progress: Whether to display a progres bar
        :param update_period: interval in seconds between different updates to
            and from the coordinator

        """
        super().__init__(
            coordinator=coordinator,
            update_period=update_period,
            worker_id=worker_id,
            progress=progress,
        )
        self.u = u
        self.num_samples = len(self.u.data)
        self.pbar = maybe_progress(
            self.num_samples,
            self.progress,
            position=worker_id,
            desc=f"Worker {worker_id}",
        )
        self._iteration_count = 1
        self._avg_values: Optional[Union[float, "NDArray"]] = None
        self._var_values: Optional[Union[float, "NDArray"]] = None

    def _compute_values(self, *args, **kwargs) -> "NDArray":
        # Import here to avoid errors with circular imports
        from .montecarlo import _permutation_montecarlo_marginals

        return _permutation_montecarlo_marginals(self.u, max_permutations=1)[0]  # type: ignore

    def run(self, *args, **kwargs):
        """Runs the worker.

        This calls :meth:`_compute_values` a certain number of times and
        calculates Shapley values on different permutations of the indices.
        After a number of seconds equal to update_period has passed, it
        reports the results to the coordinator. Before starting the next
        iteration, it checks the is_done flag, and if true terminates.
        """
        while not self.coordinator.is_done():
            start_time = time()
            while (time() - start_time) < self.update_period:
                values = self._compute_values()
                if np.any(np.isnan(values)):
                    warnings.warn(
                        "NaN values found in model scoring. Ignoring current permutation.",
                        RuntimeWarning,
                    )
                    continue
                if self._avg_values is None:
                    self._avg_values = values
                    self._var_values = np.zeros_like(self._avg_values)
                    self._iteration_count = 1
                else:
                    self._avg_values, self._var_values = get_running_avg_variance(
                        self._avg_values,
                        self._var_values,
                        values,
                        self._iteration_count,
                    )
                    self._iteration_count += 1
            self.coordinator.add_results(
                self.worker_id,
                {
                    "values": self._avg_values,
                    "std": np.sqrt(self._var_values) / self._iteration_count ** (1 / 2),
                    "num_iter": self._iteration_count,
                },
            )

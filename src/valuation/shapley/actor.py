import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from valuation.utils import Utility, maybe_progress
from valuation.utils.config import ParallelConfig

from ..utils.parallel.actor import Coordinator, RayActorWrapper, Worker
from ..utils.parallel.backend import init_parallel_backend

if TYPE_CHECKING:
    from numpy.typing import NDArray


__all__ = ["get_shapley_coordinator", "get_shapley_worker"]


logger = logging.getLogger(__name__)


def get_shapley_coordinator(
    *args, config: ParallelConfig = ParallelConfig(), **kwargs
) -> "ShapleyCoordinator":
    parallel_backend = init_parallel_backend(config)
    if config.backend == "ray":
        remote_cls = parallel_backend.wrap(ShapleyCoordinator)
        handle = remote_cls.remote(*args, **kwargs)
        coordinator = RayActorWrapper(handle, parallel_backend)
    else:
        raise NotImplementedError(f"Unexpected parallel type {config.backend}")
    return coordinator  # type: ignore


def get_shapley_worker(
    *args, config: ParallelConfig = ParallelConfig(), **kwargs
) -> "ShapleyWorker":
    parallel_backend = init_parallel_backend(config)
    if config.backend == "ray":
        remote_cls = parallel_backend.wrap(ShapleyWorker)
        handle = remote_cls.remote(*args, **kwargs)
        worker = RayActorWrapper(handle, parallel_backend)
    else:
        raise NotImplementedError(f"Unexpected parallel type {config.backend}")
    return worker  # type: ignore


class ShapleyCoordinator(Coordinator):
    def __init__(
        self,
        score_tolerance: Optional[float] = None,
        max_iterations: Optional[int] = None,
        progress: Optional[bool] = True,
    ):
        """
         The coordinator has two main tasks: aggregating the results of the workers
         and terminating the process once a certain accuracy or total number of
         iterations is reached.

        :param score_tolerance: During calculation of shapley values, the
             coordinator will check if the median standard deviation over average
             score for each point's has dropped below score_tolerance.
             If so, the computation will be terminated.
         :param max_iterations: a sum of the total number of permutation is calculated
             If the current number of permutations has exceeded max_iterations, computation
             will stop.
         :param progress: True to plot progress, False otherwise.
        """
        super().__init__(progress=progress)
        if score_tolerance is None and max_iterations is None:
            raise ValueError(
                "At least one between score_tolerance and max_iterations must be passed,"
                "or the process cannot be stopped."
            )
        self.score_tolerance = score_tolerance
        self.max_iterations = max_iterations

    def get_results(self):
        """
        It aggregates the results of the different workers and returns
        the average and std of the values. If no worker has reported yet,
        it returns two empty arrays
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
            self._total_iterations += result["num_iter"]

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

    def check_status(self):
        """
        It checks whether the accuracy of the calculation or the total number of iterations have crossed
        the set thresholds.
        If so, it sets the is_done label as True.
        """
        if len(self.workers_results) == 0:
            logger.info("No worker has updated its status yet.")
            self._is_done = False
        else:
            value, std = self.get_results()
            std_to_val_ratio = np.median(std) / np.median(value)
            if (
                self.score_tolerance is not None
                and std_to_val_ratio < self.score_tolerance
            ):
                self._is_done = True
            if (
                self.max_iterations is not None
                and self._total_iterations > self.max_iterations
            ):
                self._is_done = True
        return self._is_done


class ShapleyWorker(Worker):
    """A worker. It should work."""

    def __init__(
        self,
        u: Utility,
        coordinator: ShapleyCoordinator,
        worker_id: int,
        *,
        update_frequency: int = 30,
        progress: bool = False,
    ):
        """
        The workers calculate the Shapley values using the permutation
        definition and report the results to the coordinator.

        :param u: Utility object with model, data, and scoring function
        :param coordinator: worker results will be pushed to this coordinator
        :param worker_id: id used for reporting through maybe_progress
        :param progress: set to True to report progress, else False
        :param update_frequency: interval in seconds among different updates to
            and from the coordinator
        """
        super().__init__(
            coordinator=coordinator,
            update_frequency=update_frequency,
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

    def _compute_values(self, *args, **kwargs) -> "NDArray":
        # Importing it here avoids errors with circular imports
        from .montecarlo import _permutation_montecarlo_shapley

        return _permutation_montecarlo_shapley(self.u, max_permutations=1)[0]  # type: ignore

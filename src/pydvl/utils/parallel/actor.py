import abc
import inspect
import logging
import warnings
from time import time
from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np
from ray import ObjectRef

from ..numeric import get_running_avg_variance
from .backend import RayParallelBackend

if TYPE_CHECKING:
    from numpy.typing import NDArray


__all__ = ["RayActorWrapper", "Coordinator", "Worker"]


logger = logging.getLogger(__name__)


class RayActorWrapper:
    """Taken almost verbatim from:
    https://github.com/JaneliaSciComp/ray-janelia/blob/main/remote_as_local_wrapper.py

    :Example:

    >>> from pydvl.utils.parallel import init_parallel_backend
    >>> from pydvl.utils.config import ParallelConfig
    >>> from pydvl.utils.parallel.actor import RayActorWrapper
    >>> class Actor:
    ...     def __init__(self, x):
    ...         self.x = x
    ...
    ...     def get(self):
    ...         return self.x
    ...
    >>> config = ParallelConfig()
    >>> parallel_backend = init_parallel_backend(config)
    >>> actor_handle = parallel_backend.wrap(Actor).remote(5)
    >>> parallel_backend.get(actor_handle.get.remote())
    5
    >>> wrapped_actor = RayActorWrapper(actor_handle, parallel_backend)
    >>> wrapped_actor.get()
    5
    """

    def __init__(self, actor_handle: ObjectRef, parallel_backend: RayParallelBackend):
        self.actor_handle = actor_handle

        def remote_caller(method_name: str):
            # Wrapper for remote class's methods to mimic local calls
            def wrapper(*args, block: bool = True, **kwargs):
                obj_ref = getattr(self.actor_handle, method_name).remote(
                    *args, **kwargs
                )
                if block:
                    return parallel_backend.get(
                        obj_ref, timeout=300
                    )  # Block until called method returns.
                else:
                    return obj_ref  # Don't block and return a future.

            return wrapper

        for member in inspect.getmembers(self.actor_handle):
            name = member[0]
            if not name.startswith("__"):
                # Wrap public methods for remote-as-local calls.
                setattr(self, name, remote_caller(name))


class Coordinator(abc.ABC):
    def __init__(
        self,
        *,
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
        self.progress = progress
        self.workers_results: Dict[int, Dict[str, float]] = dict()
        self._total_iterations = 0
        self._is_done = False

    def add_results(self, worker_id: int, results: Dict):
        """
        Used by workers to report their results. It puts the results
        directly into the worker_status dictionary.

        :param worker_id: id of the worker
        :param results: results of worker calculations
        """
        self.workers_results[worker_id] = results

    # this should be a @property, but with it ray.get messes up
    def is_done(self) -> bool:
        """
        Used by workers to check whether to terminate the process.
        It returns a flag which is True when the processes must be terminated,
        False otherwise.
        """
        return self._is_done

    @abc.abstractmethod
    def get_results(self) -> dict:
        """
        It aggregates the results of the different workers and returns
        the average and std of the values. If no worker has reported yet,
        it returns two empty arrays
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def check_status(self) -> bool:
        """
        It checks whether the accuracy of the calculation or the total number of iterations have crossed
        the set thresholds.
        If so, it sets the is_done label as True.
        """
        raise NotImplementedError()


class Worker(abc.ABC):
    """A worker. It should work."""

    def __init__(
        self,
        coordinator: "Coordinator",
        worker_id: int,
        *,
        progress: bool = False,
        update_frequency: int = 30,
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
        super().__init__()
        self.worker_id = worker_id
        self.coordinator = coordinator
        self.update_frequency = update_frequency
        self.progress = progress
        self._iteration_count = 1
        self._avg_values: Optional[Union[float, "NDArray"]] = None
        self._var_values: Optional[Union[float, "NDArray"]] = None

    @abc.abstractmethod
    def _compute_values(self, *args, **kwargs) -> "NDArray":
        raise NotImplementedError()

    def run(self, *args, **kwargs):
        """Runs the worker.
        It calls _permutation_montecarlo_shapley a certain number of times and calculates
        Shapley values on different permutations of the indices.
        After a number of seconds equal to update_frequency has passed, it reports the results
        to the coordinator. Before starting the next iteration, it checks the is_done flag, and if true
        terminates.
        """
        while not self.coordinator.is_done():
            start_time = time()
            while (time() - start_time) < self.update_frequency:
                values = self._compute_values()
                if np.any(np.isnan(values)):
                    warnings.warn(
                        "Nan values found in model scoring. Ignoring current permutation.",
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

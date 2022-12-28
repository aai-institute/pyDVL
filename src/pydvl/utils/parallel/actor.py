import abc
import inspect
import logging
from typing import Any, Dict, Optional, Union

from ray import ObjectRef

from .backend import RayParallelBackend

__all__ = ["RayActorWrapper", "Coordinator", "Worker"]


logger = logging.getLogger(__name__)


class RayActorWrapper:
    """Wrapper to call methods of remote Ray actors as if they were local.

    Taken almost verbatim from:
    https://github.com/JaneliaSciComp/ray-janelia/blob/main/remote_as_local_wrapper.py

    :Example:

    >>> from pydvl.utils.parallel.backend import RayParallelBackend, init_parallel_backend
    >>> from pydvl.utils.config import ParallelConfig
    >>> from pydvl.utils.parallel.actor import RayActorWrapper
    >>> class Actor:
    ...     def __init__(self, x):
    ...         self.x = x
    ...
    ...     def get(self):
    ...         return self.x
    ...
    >>> config = ParallelConfig(backend="ray")
    >>> parallel_backend = init_parallel_backend(config)
    >>> assert isinstance(parallel_backend, RayParallelBackend)
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
            # Wrapper for remote class' methods to mimic local calls
            def wrapper(
                *args, block: bool = True, timeout: Optional[float] = None, **kwargs
            ):
                obj_ref = getattr(self.actor_handle, method_name).remote(
                    *args, **kwargs
                )
                if block:
                    return parallel_backend.get(
                        obj_ref, timeout=timeout
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
    """The coordinator has two main tasks: aggregating the results of the
    workers and terminating the process once a certain accuracy or total
    number of iterations is reached.

    :param progress: Whether to display a progress bar
    """

    def __init__(self, *, progress: Optional[bool] = True):
        self.progress = progress
        # For each worker: values, stddev, num_iterations
        self.workers_results: Dict[int, Dict[str, float]] = dict()
        self._total_iterations = 0
        self._is_done = False

    def add_results(self, worker_id: int, results: Dict[str, Union[float, int]]):
        """Used by workers to report their results. Stores the results directly
        into the `worker_status` dictionary.

        :param worker_id: id of the worker
        :param results: results of worker calculations
        """
        self.workers_results[worker_id] = results

    # this should be a @property, but with it ray.get messes up
    def is_done(self) -> bool:
        """Used by workers to check whether to terminate their process.

        :return: `True` if workers must terminate, `False` otherwise.
        """
        return self._is_done

    @abc.abstractmethod
    def get_results(self) -> Any:
        """Aggregates the results of the different workers."""
        raise NotImplementedError()

    @abc.abstractmethod
    def check_done(self) -> bool:
        """Checks whether the accuracy of the calculation or the total number
        of iterations have crossed the set thresholds.

        If so, it sets the `is_done` label to `True`.
        """
        raise NotImplementedError()


class Worker(abc.ABC):
    """A worker, it should work."""

    def __init__(
        self,
        coordinator: "Coordinator",
        worker_id: int,
        *,
        progress: bool = False,
        update_period: int = 30,
    ):
        """A worker

        :param coordinator: worker results will be pushed to this coordinator
        :param worker_id: id used for reporting through maybe_progress
        :param progress: set to True to report progress, else False
        :param update_period: interval in seconds between different updates
            to and from the coordinator
        """
        super().__init__()
        self.worker_id = worker_id
        self.coordinator = coordinator
        self.update_period = update_period
        self.progress = progress

    def run(self, *args, **kwargs):
        """Runs the worker."""
        raise NotImplementedError()

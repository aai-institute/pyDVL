import abc
import inspect
import logging
from typing import Any, Generic, List, Optional, Protocol, Type, TypeVar, cast

from ..config import ParallelConfig
from ..status import Status
from .backend import RayParallelBackend, init_parallel_backend

__all__ = ["RayActorWrapper", "Coordinator", "Worker"]


logger = logging.getLogger(__name__)

Result = TypeVar("Result")  # Avoids circular import with ValuationResult


class QueueType(Protocol):
    def put(
        self, item: Any, block: bool = True, timeout: Optional[float] = None
    ) -> None:
        ...

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        ...

    def empty(self) -> bool:
        ...

    def shutdown(self, force: bool = False, grace_period_s: int = 5) -> None:
        ...


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
    >>> wrapped_actor = RayActorWrapper(Actor, config, 5)
    >>> wrapped_actor.get()
    5
    """

    def __init__(self, actor_class: Type, config: ParallelConfig, *args, **kwargs):
        parallel_backend = cast(RayParallelBackend, init_parallel_backend(config))
        remote_cls = parallel_backend.wrap(actor_class)
        self.actor_handle = remote_cls(*args, **kwargs)

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

        for member in inspect.getmembers(actor_class):
            name = member[0]
            if not name.startswith("__"):
                # Wrap public methods for remote-as-local calls.
                setattr(self, name, remote_caller(name))


class Coordinator(Generic[Result], abc.ABC):
    """The coordinator has two main tasks: aggregating the results of the
    workers and terminating the process once a certain accuracy or total
    number of iterations is reached.

    :param queue: Used by workers to report their results to the coordinator.
    """

    _status: Status

    def __init__(self, queue: QueueType):
        self.queue = queue
        self.result: Result
        self._status = Status.Pending

    # this should be a @property, but with it ray.get messes up
    def is_done(self) -> bool:
        """Used by workers to check whether to terminate their process.

        :return: ``True`` if workers must terminate, ``False`` otherwise.
        """
        return bool(self._status)

    @abc.abstractmethod
    def check_convergence(self) -> bool:
        """Evaluates the convergence criteria on the aggregated results."""
        raise NotImplementedError()

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """Runs the coordinator."""
        raise NotImplementedError()


class Worker(abc.ABC):
    """Abstract worker class for use with TMCS.

    :param queue: Used by workers to report their results to the coordinator.
    :param worker_id: id used for reporting through maybe_progress.
    :param update_period: interval in seconds between different updates
        to and from the coordinator.
    """

    def __init__(
        self,
        queue: QueueType,
        worker_id: int,
        *,
        update_period: int = 30,
    ):
        super().__init__()
        self.queue = queue
        self.worker_id = worker_id
        self.update_period = update_period

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """Runs the worker."""
        raise NotImplementedError()

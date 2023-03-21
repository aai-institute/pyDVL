import logging
import queue
import sys
import threading
import time
import types
from concurrent.futures import BrokenExecutor, Executor, Future
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Callable, Generator, Optional, TypeVar
from weakref import ref

import ray

from ..config import ParallelConfig

__all__ = ["init_executor", "RayExecutor"]

T = TypeVar("T")

logger = logging.getLogger(__name__)


@contextmanager
def init_executor(
    config: ParallelConfig = ParallelConfig(),
) -> Generator[Executor, None, None]:
    """Initializes a futures executor based on the passed parallel configuration object.

    :param max_workers: Maximum number of concurrent tasks.
    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig` with cluster address, number of cpus, etc.

    :Example:

    >>> from pydvl.utils.parallel.futures import init_executor
    >>> from pydvl.utils.config import ParallelConfig
    >>> config = ParallelConfig(backend="ray")
    >>> with init_executor(config=config) as executor:
    ...     pass

    """
    if config.backend == "ray":
        max_workers = config.n_workers
        with RayExecutor(max_workers, config=config) as executor:
            yield executor
    else:
        raise NotImplementedError(f"Unexpected parallel type {config.backend}")


class RayExecutor(Executor):
    """Asynchronous executor using Ray that implements the concurrent.futures API.

    :param max_workers: Maximum number of concurrent tasks.
    :param config: instance of :class:`~pydvl.utils.config.ParallelConfig` with cluster address, number of cpus, etc.

    Example:
        >>> from pydvl.utils.parallel.futures import RayExecutor
        >>> with RayExecutor() as executor:
        ...     future = executor.submit(lambda x: x + 1, 1)
        ...     result = future.result()
        ...
        >>> print(result)
        2

        >>> from pydvl.utils.parallel.futures import RayExecutor
        >>> with RayExecutor() as executor:
        ...     results = list(executor.map(lambda x: x + 1, range(5)))
        ...
        >>> print(results)
        [1, 2, 3, 4, 5]
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        *,
        config: ParallelConfig = ParallelConfig(),
    ):
        if config.backend != "ray":
            raise ValueError(
                f"Parallel backend must be set to 'ray' and not {config.backend}"
            )
        if max_workers is not None:
            if max_workers <= 0:
                raise ValueError("max_workers must be greater than 0")
            max_workers = max_workers

        config_dict = asdict(config)
        config_dict.pop("backend")
        config_dict.pop("n_workers")
        if "address" not in config_dict:
            config_dict["num_cpus"] = max_workers
        self.config = config_dict
        if not ray.is_initialized():
            ray.init(**self.config)

        self._max_workers = max_workers
        if self._max_workers is None:
            self._max_workers = int(ray._private.state.cluster_resources()["CPU"])

        self._broken = False
        self._shutdown = False
        self._cancel_pending_futures = False
        self._shutdown_lock = threading.Lock()
        self._queue_lock = threading.Lock()
        self._work_queue = queue.Queue(maxsize=self._max_workers)
        self._pending_queue = queue.SimpleQueue()

        # Work Item Manager Thread
        self._work_item_manager_thread: Optional[_WorkItemManagerThread] = None

    def submit(self, fn: Callable[..., T], *args, **kwargs) -> "Future[T]":
        r"""Submits a callable to be executed with the given arguments.

        Schedules the callable to be executed as fn(\*args, \**kwargs)
        and returns a Future instance representing the execution of the callable.

        :param fn: Callable.
        :param args: Positional arguments that will be passed to `fn`.
        :param kwargs: Keyword arguments that will be passed to `fn`.
        :return: A Future representing the given call.
        """
        with self._shutdown_lock:
            logger.debug("executor acquired shutdown lock")
            if self._broken:
                raise BrokenExecutor(self._broken)
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            logging.debug("Creating future and putting work item in work queue")
            future = Future()
            w = _WorkItem(future, fn, args, kwargs)
            self._put_work_item_in_queue(w)
            # We delay starting the thread until the first call to submit
            self._start_work_item_manager_thread()
            return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        logger.debug("executor shutting down")
        with self._shutdown_lock:
            logger.debug("executor acquired shutdown lock")
            self._shutdown = True
            self._cancel_pending_futures = cancel_futures

        if wait:
            logger.debug("executor waiting for futures to finish")
            # Putting None in the queue to signal
            # to work item manager thread that we are shutting down
            self._put_work_item_in_queue(None)
            logger.debug("executor waiting for work item manager thread to terminate")
            self._work_item_manager_thread.join()
            # To reduce the risk of opening too many files, remove references to
            # objects that use file descriptors.
            self._work_item_manager_thread = None
            self._work_queue = None
            self._pending_queue = None

    def _put_work_item_in_queue(self, work_item: Optional["_WorkItem"]) -> None:
        with self._queue_lock:
            logger.debug("executor acquired queue lock")
            try:
                self._work_queue.put_nowait(work_item)
            except queue.Full:
                self._pending_queue.put_nowait(work_item)

    def _start_work_item_manager_thread(self) -> None:
        if self._work_item_manager_thread is None:
            self._work_item_manager_thread = _WorkItemManagerThread(self)
            self._work_item_manager_thread.start()


class _WorkItem:
    """Inspired by code from: concurrent.futures.thread"""

    def __init__(self, future: Future, fn: Callable, args: Any, kwargs: Any):
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        if not self.future.set_running_or_notify_cancel():
            return

        remote_fn = ray.remote(self.fn)
        # TODO: we assign 1 CPU to each task which could be wasteful.
        ref = remote_fn.options(name=self.fn.__name__, num_cpus=1).remote(
            *self.args, **self.kwargs
        )

        # Almost verbatim copy of `future()` method of ClientObjectRef
        def set_future(data: Any) -> None:
            """Schedules a callback to set the exception or result
            in the Future."""

            if isinstance(data, Exception):
                self.future.set_exception(data)
            else:
                self.future.set_result(data)

        ref._on_completed(set_future)
        # Prevent this object ref from being released.
        self.future.object_ref = ref

    if sys.version_info >= (3, 9):
        __class_getitem__ = classmethod(types.GenericAlias)


class _WorkItemManagerThread(threading.Thread):
    """Manages submitting the work items and throttling.

    It runs in a local thread.

    :param executor: An instance of RayExecutor that owns
        this thread. A weakref will be owned by the manager as well as
        references to internal objects used to introspect the state of
        the executor.
    """

    def __init__(self, executor: RayExecutor):
        self.executor_reference = ref(executor)
        self.shutdown_lock: threading.Lock = executor._shutdown_lock
        self.queue_lock: threading.Lock = executor._queue_lock
        self.work_queue: Optional["queue.Queue[_WorkItem]"] = executor._work_queue
        self.pending_queue: Optional[
            "queue.SimpleQueue[_WorkItem]"
        ] = executor._pending_queue
        super().__init__()

    def run(self) -> None:
        logger.debug("starting work item manager thread main loop")
        while True:
            time.sleep(0.1)
            try:
                self.add_pending_item_to_work_queue()
                self.submit_work_item()
            finally:
                if self.is_shutting_down():
                    self.flag_executor_shutting_down()

                    # Since no new work items can be added,
                    # it is safe to shut down this thread
                    # if there are no more work items.
                    if self.work_queue.empty() and self.pending_queue.empty():
                        break
        logger.debug("exiting work item manager thread main loop")

    def add_pending_item_to_work_queue(self) -> None:
        # Fills work_queue with _WorkItems from pending_queue.
        # This function never blocks.
        while True:
            with self.queue_lock:
                logger.debug("work item manager thread acquired queue lock")
                # If the work queue is not full,
                # Move a work item from the pending queue, if not empty,
                # to the work queue
                if self.work_queue.full():
                    return
                try:
                    work_item = self.pending_queue.get_nowait()
                except queue.Empty:
                    logger.debug("pending queue is empty")
                    return
                else:
                    if work_item is None:
                        self.work_queue.put_nowait(work_item)
                        return
                    logger.debug("moving work item from pending queue to work queue")
                    self.work_queue.put(work_item)
                    del work_item

    def submit_work_item(self) -> None:
        with self.queue_lock:
            logger.debug("work item manager thread acquired queue lock")
            # Try to get a work item
            # If it is None, we break from the loop.
            # Otherwise, we submit the future.
            try:
                logger.debug("getting work item from work queue")
                work_item = self.work_queue.get_nowait()
                if work_item is None:
                    return
            except queue.Empty:
                logger.debug("work queue is empty")
                return
        logger.debug("Submitting work item")
        work_item.run()
        # Delete references to object
        del work_item

    def is_shutting_down(self) -> bool:
        # Check whether we should start shutting down the executor.
        executor = self.executor_reference()
        # No more work items can be added if:
        #   - The executor that owns this worker has been collected OR
        #   - The executor that owns this worker has been shutdown.
        with self.shutdown_lock:
            logger.debug("work item manager thread acquired shutdown lock")
            if executor is None or executor._shutdown:
                # Flag the executor as shutting down as early as possible if it
                # is not gc-ed yet.
                if executor is not None:
                    executor._shutdown = True
                return True
            return False

    def flag_executor_shutting_down(self):
        # Flag the executor as shutting down and cancel remaining tasks if
        # requested as early as possible if it is not gc-ed yet.
        executor = self.executor_reference()
        with self.shutdown_lock:
            logger.debug("work item manager thread acquired shutdown lock")
            if executor is not None:
                executor._shutdown = True
                # Cancel pending work items if requested.
                if executor._cancel_pending_futures:
                    logger.debug("forcefully cancelling futures")
                    # Drain all work items from the queues,
                    # and then cancel their associated futures.
                    # We empty the pending queue first.
                    while True:
                        with self.queue_lock:
                            try:
                                work_item = self.pending_queue.get_nowait()
                            except queue.Empty:
                                break
                            if work_item is not None:
                                work_item.future.cancel()
                                del work_item
                    while True:
                        with self.queue_lock:
                            try:
                                work_item = self.work_queue.get_nowait()
                            except queue.Empty:
                                break
                            if work_item is not None:
                                work_item.future.cancel()
                                del work_item
                    # Make sure we do this only once to not waste time looping
                    # on running processes over and over.
                    executor._cancel_pending_futures = False

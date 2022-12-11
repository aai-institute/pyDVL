import weakref
from itertools import accumulate, chain
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import ray
from ray import ObjectRef

from ..config import ParallelConfig
from ..utility import Utility
from .backend import init_parallel_backend

__all__ = ["MapReduceJob"]

from ..types import maybe_add_argument

T = TypeVar("T")
R = TypeVar("R")
Identity = lambda x, *args, **kwargs: x

MapFunction = Callable[..., R]
ReduceFunction = Callable[[Iterable[R]], R]


def wrap_func_with_remote_args(func, *, timeout: Optional[float] = None):
    def wrapper(*args, **kwargs):
        args = list(args)
        for i, v in enumerate(args[:]):
            args[i] = get_value(v, timeout=timeout)
        for k, v in kwargs.items():
            kwargs[k] = get_value(v, timeout=timeout)
        return func(*args, **kwargs)

    return wrapper


def get_value(
    v: Union[ObjectRef, Iterable[ObjectRef], Any], *, timeout: Optional[float] = None
):
    if isinstance(v, ObjectRef):
        return ray.get(v, timeout=timeout)
    elif isinstance(v, Iterable):
        return [get_value(x, timeout=timeout) for x in v]
    else:
        return v


class MapReduceJob(Generic[T, R]):
    """Takes an embarrassingly parallel fun and runs it in `n_jobs` parallel
    jobs, splitting the data into the same number of chunks, one for each job.

    It repeats the process `n_runs` times, allocating jobs across runs. E.g.
    if `n_jobs=90` and `n_runs=10`, each whole execution of fun uses 9 jobs,
    with the data split evenly among them. If `n_jobs=2` and `n_runs=10`, two
    jobs are used, five times in succession, and each job receives all data.

    Results are aggregated per run using `reduce_func`, but **not across runs**.
    A list of length `n_runs` is always returned.

    Typing information for objects of this class requires the type of the inputs
    that are split for `map_func` and the type of its output.

    :param map_func: Function that will be applied to the input chunks in each
        job.
    :param reduce_func: Function that will be applied to the results of
        `map_func` to reduce them. This will be done independently for each run,
        i.e. the reducer need and must not account for data of multiple runs.
    :param map_kwargs: Keyword arguments that will be passed to `map_func` in
        each job. Alternatively, one can use `itertools.partial`.
    :param reduce_kwargs: Keyword arguments that will be passed to `reduce_func`
        in each job. Alternatively, one can use `itertools.partial`.
    :param config: Instance of :class:`~pydvl.utils.config.ParallelConfig`
        with cluster address, number of cpus, etc.
    :param n_jobs: Number of parallel jobs to run. Does not accept 0
    :param n_runs: Number of times to run `map_func` and `reduce_func` on the
        whole data.
    :param timeout: Amount of time in seconds to wait for remote results before
        ... TODO
    :param max_parallel_tasks: Maximum number of jobs to start in parallel. Any
        tasks above this number won't be submitted to the backend before some
        are done. This is to avoid swamping the work queue. Note that tasks have
        a low memory footprint, so this is probably not a big concernt, except
        in the case of an infinite stream (not the case for MapReduceJob). See
        https://docs.ray.io/en/latest/ray-core/patterns/limit-pending-tasks.html

    :Examples:

    A simple usage example with 2 jobs and 3 runs:

    >>> from pydvl.utils.parallel import MapReduceJob
    >>> import numpy as np
    >>> map_reduce_job: MapReduceJob[np.ndarray, np.ndarray] = MapReduceJob(
    ...     map_func=np.sum,
    ...     reduce_func=np.sum,
    ...     n_jobs=2,
    ...     n_runs=3,
    ... )
    >>> map_reduce_job(np.arange(5))
    [10, 10, 10]

    """

    def __init__(
        self,
        map_func: MapFunction[R],
        reduce_func: Optional[ReduceFunction[R]] = None,
        map_kwargs: Optional[Dict] = None,
        reduce_kwargs: Optional[Dict] = None,
        config: ParallelConfig = ParallelConfig(),
        *,
        n_jobs: int = 1,
        n_runs: int = 1,
        timeout: Optional[float] = None,
        max_parallel_tasks: Optional[int] = None,
    ):
        self.config = config
        parallel_backend = init_parallel_backend(self.config)
        self._parallel_backend_ref = weakref.ref(parallel_backend)

        self.timeout = timeout
        self.n_runs = n_runs

        self._n_jobs = 1
        # This uses the setter defined below
        self.n_jobs = n_jobs

        if max_parallel_tasks is None:
            # TODO: Find a better default value?
            self.max_parallel_tasks = 2 * (self.n_jobs + self.n_runs)
        else:
            self.max_parallel_tasks = max_parallel_tasks

        if reduce_func is None:
            reduce_func = Identity

        self.map_kwargs = map_kwargs
        self.reduce_kwargs = reduce_kwargs

        if self.map_kwargs is None:
            self.map_kwargs = dict()

        if self.reduce_kwargs is None:
            self.reduce_kwargs = dict()

        self._map_func = maybe_add_argument(map_func, "job_id")
        self._reduce_func = reduce_func

    def __call__(
        self,
        inputs: Union[Sequence[T], Sequence[Utility], Utility],
    ) -> List[R]:
        if isinstance(inputs, Utility):
            inputs_ = self.parallel_backend.put(inputs)
        elif len(inputs) > 0 and isinstance(inputs[0], Utility):
            inputs_ = [self.parallel_backend.put(x) for x in inputs]
        else:
            inputs_ = inputs
        map_results = self.map(inputs_)
        reduce_results = self.reduce(map_results)
        return reduce_results

    def map(
        self, inputs: Union[Sequence[T], Sequence["ObjectRef"], "ObjectRef"]
    ) -> List[List["ObjectRef[R]"]]:
        map_results: List[List["ObjectRef[R]"]] = []

        map_func = self._wrap_function(self._map_func)

        total_n_jobs = 0
        total_n_finished = 0

        for _ in range(self.n_runs):
            # In this first case we don't use chunking at all
            if self.n_runs >= self.n_jobs:
                chunks = [inputs]
            else:
                chunks = self._chunkify(inputs, n_chunks=self.n_jobs)

            map_result = []
            for j, next_chunk in enumerate(chunks):
                result = map_func(next_chunk, job_id=j, **self.map_kwargs)
                map_result.append(result)
                total_n_jobs += 1

                total_n_finished = self._backpressure(
                    list(chain.from_iterable([*map_results, map_result])),
                    n_dispatched=total_n_jobs,
                    n_finished=total_n_finished,
                )

            map_results.append(map_result)

        return map_results

    def reduce(self, chunks: List[List["ObjectRef[R]"]]) -> List[R]:
        reduce_func = self._wrap_function(self._reduce_func)

        total_n_jobs = 0
        total_n_finished = 0
        reduce_results = []

        for i in range(self.n_runs):
            result = reduce_func(chunks[i], **self.reduce_kwargs)
            reduce_results.append(result)
            total_n_jobs += 1
            total_n_finished = self._backpressure(
                reduce_results, n_dispatched=total_n_jobs, n_finished=total_n_finished
            )

        results = self.parallel_backend.get(reduce_results, timeout=self.timeout)
        return results  # type: ignore

    def _wrap_function(self, func):
        remote_func = self.parallel_backend.wrap(
            wrap_func_with_remote_args(func, timeout=self.timeout)
        )
        if hasattr(remote_func, "remote"):
            return remote_func.remote
        else:
            return remote_func

    def _backpressure(
        self, jobs: List[ObjectRef], n_dispatched: int, n_finished: int
    ) -> int:
        """
        See https://docs.ray.io/en/latest/ray-core/patterns/limit-pending-tasks.html
        :param jobs:
        :param n_dispatched:
        :param n_finished:
        :return:
        """
        while (n_in_flight := n_dispatched - n_finished) > self.max_parallel_tasks:
            wait_for_num_jobs = n_in_flight - self.max_parallel_tasks
            finished_jobs, _ = self.parallel_backend.wait(
                jobs, num_returns=wait_for_num_jobs, timeout=10  # FIXME make parameter?
            )
            n_finished += len(finished_jobs)
        return n_finished

    @staticmethod
    def _chunkify(data: Sequence[T], n_chunks: int) -> Iterator[Sequence[T]]:
        # Splits a list of values into chunks for each job
        if n_chunks == 0:
            raise ValueError("Number of chunks should be greater than 0")

        elif n_chunks == 1:
            yield data

        else:
            n = len(data)

            # This is very much inspired by numpy's array_split function
            # The difference is that it only uses built-in functions
            # and does not convert the input data to an array
            chunk_size, remainder = divmod(n, n_chunks)
            chunk_indices = tuple(
                accumulate(
                    [0]
                    + remainder * [chunk_size + 1]
                    + (n_chunks - remainder) * [chunk_size]
                )
            )
            for start_index, end_index in zip(chunk_indices[:-1], chunk_indices[1:]):
                if start_index >= end_index:
                    return
                yield data[start_index:end_index]

    @property
    def parallel_backend(self):
        parallel_backend = self._parallel_backend_ref()
        if parallel_backend is None:
            raise RuntimeError(f"Could not get reference to parallel backend instance")
        return parallel_backend

    @property
    def n_jobs(self) -> int:
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value: int):
        self._n_jobs = self.parallel_backend.effective_n_jobs(value)

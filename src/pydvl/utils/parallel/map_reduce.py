from collections.abc import Iterable, Sequence
from functools import singledispatch
from itertools import accumulate, chain, repeat
from typing import Any, Callable, Dict, Generic
from typing import Iterable as IterableType
from typing import List, Optional
from typing import Sequence as SequenceType
from typing import TypeVar, Union

import ray
from ray import ObjectRef

from ..config import ParallelConfig
from ..types import maybe_add_argument
from .backend import init_parallel_backend

__all__ = ["MapReduceJob"]

T = TypeVar("T")
R = TypeVar("R")
Identity = lambda x, *args, **kwargs: x

MapFunction = Callable[..., R]
ReduceFunction = Callable[[IterableType[R]], R]


def _wrap_func_with_remote_args(func: Callable, *, timeout: Optional[float] = None):
    def wrapper(*args, **kwargs):
        args = list(args)
        for i, v in enumerate(args[:]):
            args[i] = _get_value(v, timeout=timeout)
        for k, v in kwargs.items():
            kwargs[k] = _get_value(v, timeout=timeout)
        return func(*args, **kwargs)

    # Doing it manually here because using wraps or update_wrapper
    # from functools doesn't work with ray for some unknown reason
    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__doc__ = func.__doc__
    return wrapper


@singledispatch
def _get_value(v: Any, *, timeout: Optional[float] = None) -> Any:
    return v


@_get_value.register
def _(v: ObjectRef, *, timeout: Optional[float] = None) -> Any:
    return ray.get(v, timeout=timeout)


@_get_value.register
def _(v: Iterable, *, timeout: Optional[float] = None) -> List[Any]:
    return [_get_value(x, timeout=timeout) for x in v]


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

    :param inputs:
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
    ...     np.arange(5),
    ...     map_func=np.sum,
    ...     reduce_func=np.sum,
    ...     n_jobs=2,
    ...     n_runs=3,
    ... )
    >>> map_reduce_job()
    [10, 10, 10]

    When passed a single object as input, it will be repeated for each job, if n_jobs > n_runs:

    >>> from pydvl.utils.parallel import MapReduceJob
    >>> import numpy as np
    >>> map_reduce_job: MapReduceJob[int, np.ndarray] = MapReduceJob(
    ...     5,
    ...     map_func=np.sum,
    ...     reduce_func=np.sum,
    ...     n_jobs=4,
    ...     n_runs=2,
    ... )
    >>> map_reduce_job()
    [20, 20]
    """

    def __init__(
        self,
        inputs: Union[SequenceType[T], T],
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
        self.parallel_backend = parallel_backend

        self.timeout = timeout
        self.n_runs = n_runs

        self._n_jobs = 1
        # This uses the setter defined below
        self.n_jobs = n_jobs

        # TODO: Find a better default value?
        default_max_parallel_tasks = 2 * (self.n_jobs + self.n_runs)
        self.max_parallel_tasks = max_parallel_tasks or default_max_parallel_tasks

        self.inputs_ = inputs

        if reduce_func is None:
            reduce_func = Identity

        if map_kwargs is None:
            self.map_kwargs = dict()
        else:
            self.map_kwargs = {
                k: self.parallel_backend.put(v) for k, v in map_kwargs.items()
            }

        if reduce_kwargs is None:
            self.reduce_kwargs = dict()
        else:
            self.reduce_kwargs = {
                k: self.parallel_backend.put(v) for k, v in reduce_kwargs.items()
            }

        self._map_func = maybe_add_argument(map_func, "job_id")
        self._reduce_func = reduce_func

    def __call__(
        self,
    ) -> List[R]:
        map_results = self.map(self.inputs_)
        reduce_results = self.reduce(map_results)
        return reduce_results

    def map(
        self, inputs: Union[SequenceType[T], "ObjectRef[T]"]
    ) -> List[List["ObjectRef[R]"]]:
        map_results: List[List["ObjectRef[R]"]] = []

        map_func = self._wrap_function(self._map_func)

        total_n_jobs = 0
        total_n_finished = 0

        # In the first case we don't use chunking at all
        if self.n_runs >= self.n_jobs:
            chunks = self._chunkify(inputs, n_chunks=1)
        else:
            chunks = self._chunkify(inputs, n_chunks=self.n_jobs)

        for _ in range(self.n_runs):
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

    def _wrap_function(self, func: Callable, **kwargs) -> Callable:
        remote_func = self.parallel_backend.wrap(
            _wrap_func_with_remote_args(func, timeout=self.timeout), **kwargs
        )
        return getattr(remote_func, "remote", remote_func)  # type: ignore

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

    def _chunkify(
        self, data: Union[SequenceType[T], T], n_chunks: int
    ) -> List["ObjectRef[T]"]:
        """If data is a SequenceType, it splits it into SequenceTypes of size `n_chunks` for each job that we call chunks.
        If instead data is an `ObjectRef` instance, then it yields it repeatedly `n_chunks` number of times.
        """
        if n_chunks == 0:
            raise ValueError("Number of chunks should be greater than 0")

        elif n_chunks == 1:
            data_id = self.parallel_backend.put(data)
            return [data_id]

        elif not isinstance(data, Sequence):
            data_id = self.parallel_backend.put(data)
            return list(repeat(data_id, times=n_chunks))
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

            chunks = []

            for start_index, end_index in zip(chunk_indices[:-1], chunk_indices[1:]):
                if start_index >= end_index:
                    break
                chunk_id = self.parallel_backend.put(data[start_index:end_index])
                chunks.append(chunk_id)

            return chunks

    @property
    def n_jobs(self) -> int:
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value: int):
        self._n_jobs = self.parallel_backend.effective_n_jobs(value)

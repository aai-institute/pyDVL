from itertools import accumulate, repeat
from typing import Callable, Dict, Generic, List, Optional, Sequence, TypeVar, Union

from joblib import Parallel, delayed
from numpy.typing import NDArray
from ray.util.joblib import register_ray

from ..config import ParallelConfig
from ..types import maybe_add_argument
from .backend import init_parallel_backend

__all__ = ["MapReduceJob"]

T = TypeVar("T")
R = TypeVar("R")
Identity = lambda x, *args, **kwargs: x

MapFunction = Callable[..., R]
ReduceFunction = Callable[[List[R]], R]


register_ray()


class MapReduceJob(Generic[T, R]):
    """Takes an embarrassingly parallel fun and runs it in ``n_jobs`` parallel
    jobs, splitting the data evenly into a number of chunks equal to the number of jobs.

    Typing information for objects of this class requires the type of the inputs
    that are split for ``map_func`` and the type of its output.

    :param inputs: The input that will be split and passed to `map_func`.
        if it's not a sequence object. It will be repeat ``n_jobs`` number of times.
    :param map_func: Function that will be applied to the input chunks in each job.
    :param reduce_func: Function that will be applied to the results of
        ``map_func`` to reduce them.
    :param map_kwargs: Keyword arguments that will be passed to ``map_func`` in
        each job. Alternatively, one can use ``itertools.partial``.
    :param reduce_kwargs: Keyword arguments that will be passed to ``reduce_func``
        in each job. Alternatively, one can use :func:`itertools.partial`.
    :param config: Instance of :class:`~pydvl.utils.config.ParallelConfig`
        with cluster address, number of cpus, etc.
    :param n_jobs: Number of parallel jobs to run. Does not accept 0
    :param timeout: Amount of time in seconds to wait for remote results before
        ... TODO
    :param max_parallel_tasks: Maximum number of jobs to start in parallel. Any
        tasks above this number won't be submitted to the backend before some
        are done. This is to avoid swamping the work queue. Note that tasks have
        a low memory footprint, so this is probably not a big concern, except
        in the case of an infinite stream (not the case for MapReduceJob). See
        https://docs.ray.io/en/latest/ray-core/patterns/limit-pending-tasks.html

    :Examples:

    A simple usage example with 2 jobs:

    >>> from pydvl.utils.parallel import MapReduceJob
    >>> import numpy as np
    >>> map_reduce_job: MapReduceJob[np.ndarray, np.ndarray] = MapReduceJob(
    ...     np.arange(5),
    ...     map_func=np.sum,
    ...     reduce_func=np.sum,
    ...     n_jobs=2,
    ... )
    >>> map_reduce_job()
    10

    When passed a single object as input, it will be repeated for each job:

    >>> from pydvl.utils.parallel import MapReduceJob
    >>> import numpy as np
    >>> map_reduce_job: MapReduceJob[int, np.ndarray] = MapReduceJob(
    ...     5,
    ...     map_func=lambda x: np.array([x]),
    ...     reduce_func=np.sum,
    ...     n_jobs=4,
    ... )
    >>> map_reduce_job()
    20
    """

    def __init__(
        self,
        inputs: Union[Sequence[T], T],
        map_func: MapFunction[R],
        reduce_func: Optional[ReduceFunction[R]] = None,
        map_kwargs: Optional[Dict] = None,
        reduce_kwargs: Optional[Dict] = None,
        config: ParallelConfig = ParallelConfig(),
        *,
        n_jobs: int = -1,
        timeout: Optional[float] = None,
        max_parallel_tasks: Optional[int] = None,
    ):
        self.config = config
        parallel_backend = init_parallel_backend(self.config)
        self.parallel_backend = parallel_backend

        self.timeout = timeout

        self._n_jobs = 1
        # This uses the setter defined below
        self.n_jobs = n_jobs

        self.max_parallel_tasks = max_parallel_tasks

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
    ) -> R:
        if self.config.backend == "joblib":
            backend = "loky"
        else:
            backend = self.config.backend
        with Parallel(backend=backend, n_jobs=self.n_jobs) as parallel:
            chunks = self._chunkify(self.inputs_, n_chunks=self.n_jobs)
            map_results: List[R] = parallel(
                delayed(self._map_func)(next_chunk, job_id=j, **self.map_kwargs)
                for j, next_chunk in enumerate(chunks)
            )
        reduce_results: R = self._reduce_func(map_results, **self.reduce_kwargs)
        return reduce_results

    def _chunkify(
        self, data: Union[NDArray[T], Sequence[T], T], n_chunks: int
    ) -> List[Union[NDArray[T], Sequence[T], T]]:
        """If data is a Sequence, it splits it into Sequences of size `n_chunks` for each job that we call chunks.
        If instead data is an `ObjectRef` instance, then it yields it repeatedly `n_chunks` number of times.
        """
        if n_chunks <= 0:
            raise ValueError("Number of chunks should be greater than 0")

        if n_chunks == 1:
            return [data]

        try:
            # This is used as a check to determine whether data is iterable or not
            # if it's the former, then the value will be used to determine the chunk indices.
            n = len(data)
        except TypeError:
            return list(repeat(data, times=n_chunks))
        else:
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
                chunk = data[start_index:end_index]
                chunks.append(chunk)

            return chunks

    @property
    def n_jobs(self) -> int:
        """Effective number of jobs according to the used ParallelBackend instance."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value: int):
        self._n_jobs = self.parallel_backend.effective_n_jobs(value)

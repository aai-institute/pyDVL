"""
This module contains a wrapper around joblib's `Parallel()` class that makes it
easy to run map-reduce jobs.

!!! Deprecation notice
    This interface might be deprecated or changed in a future release before 1.0

"""
from itertools import accumulate, repeat
from typing import Any, Collection, Dict, Generic, List, Optional, TypeVar, Union

from joblib import Parallel, delayed
from numpy.typing import NDArray

from ..config import ParallelConfig
from ..types import MapFunction, ReduceFunction, maybe_add_argument
from .backend import init_parallel_backend

__all__ = ["MapReduceJob"]

T = TypeVar("T")
R = TypeVar("R")


def identity(x: Any, *args: Any, **kwargs: Any) -> Any:
    return x


class MapReduceJob(Generic[T, R]):
    """Takes an embarrassingly parallel fun and runs it in `n_jobs` parallel
    jobs, splitting the data evenly into a number of chunks equal to the number of jobs.

    Typing information for objects of this class requires the type of the inputs
    that are split for `map_func` and the type of its output.

    Args:
        inputs: The input that will be split and passed to `map_func`.
            if it's not a sequence object. It will be repeat `n_jobs` number of times.
        map_func: Function that will be applied to the input chunks in each job.
        reduce_func: Function that will be applied to the results of
            `map_func` to reduce them.
        map_kwargs: Keyword arguments that will be passed to `map_func` in
            each job. Alternatively, one can use [functools.partial][].
        reduce_kwargs: Keyword arguments that will be passed to `reduce_func`
            in each job. Alternatively, one can use [functools.partial][].
        config: Instance of [ParallelConfig][pydvl.utils.config.ParallelConfig]
            with cluster address, number of cpus, etc.
        n_jobs: Number of parallel jobs to run. Does not accept 0

    ??? Example
        A simple usage example with 2 jobs:

        ``` pycon
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
        ```

        When passed a single object as input, it will be repeated for each job:
        ``` pycon
        >>> from pydvl.utils.parallel import MapReduceJob
        >>> import numpy as np
        >>> map_reduce_job: MapReduceJob[int, np.ndarray] = MapReduceJob(
        ...     5,
        ...     map_func=lambda x: np.array([x]),
        ...     reduce_func=np.sum,
        ...     n_jobs=2,
        ... )
        >>> map_reduce_job()
        10
        ```
    """

    def __init__(
        self,
        inputs: Union[Collection[T], T],
        map_func: MapFunction[R],
        reduce_func: ReduceFunction[R] = identity,
        map_kwargs: Optional[Dict] = None,
        reduce_kwargs: Optional[Dict] = None,
        config: ParallelConfig = ParallelConfig(),
        *,
        n_jobs: int = -1,
        timeout: Optional[float] = None,
    ):
        self.config = config
        parallel_backend = init_parallel_backend(self.config)
        self.parallel_backend = parallel_backend

        self.timeout = timeout

        # This uses the setter defined below
        self.n_jobs = n_jobs

        self.inputs_ = inputs

        self.map_kwargs = map_kwargs if map_kwargs is not None else dict()
        self.reduce_kwargs = reduce_kwargs if reduce_kwargs is not None else dict()

        self._map_func = maybe_add_argument(map_func, "job_id")
        self._reduce_func = reduce_func

    def __call__(
        self,
    ) -> R:
        if self.config.backend == "joblib":
            backend = "loky"
        else:
            backend = self.config.backend
        # In joblib the levels are reversed.
        # 0 means no logging and 50 means log everything to stdout
        verbose = 50 - self.config.logging_level
        with Parallel(backend=backend, n_jobs=self.n_jobs, verbose=verbose) as parallel:
            chunks = self._chunkify(self.inputs_, n_chunks=self.n_jobs)
            map_results: List[R] = parallel(
                delayed(self._map_func)(next_chunk, job_id=j, **self.map_kwargs)
                for j, next_chunk in enumerate(chunks)
            )
        reduce_results: R = self._reduce_func(map_results, **self.reduce_kwargs)
        return reduce_results

    def _chunkify(
        self, data: Union[NDArray, Collection[T], T], n_chunks: int
    ) -> List[Union[NDArray, Collection[T], T]]:
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
            n = len(data)  # type: ignore
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
                chunk = data[start_index:end_index]  # type: ignore
                chunks.append(chunk)

            return chunks

    @property
    def n_jobs(self) -> int:
        """Effective number of jobs according to the used ParallelBackend instance."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value: int):
        self._n_jobs = self.parallel_backend.effective_n_jobs(value)

import operator
import os
import time
from functools import partial, reduce
from typing import Optional

import numpy as np
import pytest

from pydvl.parallel import MapReduceJob, RayParallelBackend
from pydvl.utils.types import Seed

from ..conftest import num_workers


def test_effective_n_jobs(parallel_backend):
    assert parallel_backend.effective_n_jobs(1) == 1
    assert parallel_backend.effective_n_jobs(4) == min(4, num_workers())
    assert parallel_backend.effective_n_jobs(-1) == num_workers()

    for n_jobs in [-1, 1, 2]:
        assert parallel_backend.effective_n_jobs(n_jobs) > 0

    with pytest.raises(ValueError):
        parallel_backend.effective_n_jobs(0)


@pytest.fixture()
def map_reduce_job_and_parameters(parallel_backend, n_jobs, request):
    try:
        kind, map_func, reduce_func = request.param
        assert kind == "custom"
    except ValueError:
        kind = request.param
    if kind == "numpy":
        map_reduce_job = partial(
            MapReduceJob,
            map_func=np.sum,
            reduce_func=np.sum,
            parallel_backend=parallel_backend,
            n_jobs=n_jobs,
        )
    elif kind == "list":
        map_reduce_job = partial(
            MapReduceJob,
            map_func=lambda x: x,
            reduce_func=lambda r: reduce(operator.add, r, []),
            parallel_backend=parallel_backend,
            n_jobs=n_jobs,
        )
    elif kind == "range":
        map_reduce_job = partial(
            MapReduceJob,
            map_func=lambda x: list(x),
            reduce_func=lambda r: reduce(operator.add, list(r), []),
            parallel_backend=parallel_backend,
            n_jobs=n_jobs,
        )
    elif kind == "custom":
        map_reduce_job = partial(
            MapReduceJob,
            map_func=map_func,
            reduce_func=reduce_func,
            parallel_backend=parallel_backend,
            n_jobs=n_jobs,
        )
    else:
        map_reduce_job = partial(
            MapReduceJob,
            map_func=lambda x: x * x,
            reduce_func=lambda r: r,
            parallel_backend=parallel_backend,
            n_jobs=n_jobs,
        )
    return map_reduce_job, n_jobs


@pytest.mark.parametrize(
    "map_reduce_job_and_parameters, indices, expected",
    [
        ("list", [], []),
        ("list", [1, 2], [1, 2]),
        ("list", [1, 2, 3, 4], [1, 2, 3, 4]),
        ("range", range(10), list(range(10))),
        ("numpy", np.arange(10), 45),
    ],
    indirect=["map_reduce_job_and_parameters"],
)
@pytest.mark.parametrize("n_jobs", [1, 2, 4])
def test_map_reduce_job(map_reduce_job_and_parameters, indices, expected):
    map_reduce_job, n_jobs = map_reduce_job_and_parameters
    result = map_reduce_job(indices)()
    assert np.all(result == expected)


@pytest.mark.parametrize(
    "data, n_chunks, expected_chunks",
    [
        ([], 3, []),
        ([1, 2, 3], 2, [[1, 2], [3]]),
        ([1, 2, 3, 4], 2, [[1, 2], [3, 4]]),
        ([1, 2, 3, 4], 3, [[1, 2], [3], [4]]),
        ([1, 2, 3, 4], 5, [[1], [2], [3], [4]]),
        (list(range(5)), 42, [[i] for i in range(5)]),
        (np.arange(5), 42, [[i] for i in range(5)]),
        (range(10), 4, [range(0, 3), range(3, 6), range(6, 8), range(8, 10)]),
        (np.arange(10), 4, np.array_split(np.arange(10), 4)),
    ],
)
def test_chunkification(parallel_backend, data, n_chunks, expected_chunks):
    map_reduce_job = MapReduceJob(
        [], map_func=lambda x: x, parallel_backend=parallel_backend
    )
    chunks = list(map_reduce_job._chunkify(data, n_chunks))
    for x, y in zip(chunks, expected_chunks):
        assert np.all(x == y)


def test_map_reduce_job_partial_map_and_reduce_func(parallel_backend):
    def map_func(x, y):
        return x + y

    def reduce_func(x, y):
        return np.sum(np.concatenate(x)) + y

    map_func = partial(map_func, y=10)
    reduce_func = partial(reduce_func, y=5)

    map_reduce_job = MapReduceJob(
        np.arange(10),
        map_func=map_func,
        reduce_func=reduce_func,
        parallel_backend=parallel_backend,
    )
    result = map_reduce_job()
    assert result == 150


@pytest.mark.parametrize(
    "seed_1, seed_2",
    [
        (42, 12),
    ],
)
def test_map_reduce_seeding(parallel_backend, seed_1, seed_2):
    """Test that the same result is obtained when using the same seed. And that
    different results are obtained when using different seeds.
    """

    def _sum_of_random_integers(x: None = None, seed: Optional[Seed] = None):
        rng = np.random.default_rng(seed)
        values = rng.integers(0, rng.integers(10, 100), 10)
        return np.sum(values)

    map_reduce_job = MapReduceJob(
        None,
        map_func=_sum_of_random_integers,
        reduce_func=np.mean,
        parallel_backend=parallel_backend,
    )
    result_1 = map_reduce_job(seed=seed_1)
    result_2 = map_reduce_job(seed=seed_1)
    result_3 = map_reduce_job(seed=seed_2)
    assert result_1 == result_2
    assert result_1 != result_3


@pytest.mark.flaky(reruns=2)  # the ways of ray are mysterious
def test_wrap_function(parallel_backend):
    if not isinstance(parallel_backend, RayParallelBackend):
        pytest.skip("Only makes sense for ray")

    def fun(x, **kwargs):
        return dict(x=x * x, **kwargs)

    # Try two kwargs for @ray.remote. Should be ignored in the joblib backend
    wrapped_func = parallel_backend.wrap(fun, num_cpus=1, max_calls=1)
    x = parallel_backend.put(2)
    ret = parallel_backend.get(wrapped_func(x))

    assert ret["x"] == 4
    assert len(ret) == 1  # Ensure that kwargs are not passed to the function

    # Test that the function is executed in different processes
    def get_pid():
        time.sleep(2)  # FIXME: waiting less means fewer processes are used?!
        return os.getpid()

    wrapped_func = parallel_backend.wrap(get_pid, num_cpus=1)
    pids = parallel_backend.get([wrapped_func() for _ in range(num_workers())])
    assert len(set(pids)) == num_workers()


def test_futures_executor_submit(parallel_backend):
    with parallel_backend.executor() as executor:
        future = executor.submit(lambda x: x + 1, 1)
        result = future.result()
    assert result == 2


def test_futures_executor_map(parallel_backend):
    with parallel_backend.executor() as executor:
        results = list(executor.map(lambda x: x + 1, range(3)))
    assert results == [1, 2, 3]


def test_futures_executor_map_with_max_workers(parallel_backend):
    def func(_):
        time.sleep(1)
        return time.monotonic()

    start_time = time.monotonic()
    with parallel_backend.executor(max_workers=num_workers()) as executor:
        assert executor._max_workers == num_workers()
        list(executor.map(func, range(3)))
    end_time = time.monotonic()
    total_time = end_time - start_time
    # We expect the time difference to be > 3 / num_workers(), but has to be at least 1
    assert total_time > max(1.0, 3 / num_workers())


@pytest.mark.timeout(30)
@pytest.mark.flaky(reruns=2)
def test_future_cancellation(parallel_backend):
    if not isinstance(parallel_backend, RayParallelBackend):
        pytest.skip("Currently this test only works with Ray")

    from pydvl.parallel import CancellationPolicy

    with parallel_backend.executor(cancel_futures=CancellationPolicy.NONE) as executor:
        future = executor.submit(lambda x: x + 1, 1)

    assert future.result() == 2

    from ray.exceptions import RayTaskError, TaskCancelledError

    with parallel_backend.executor(cancel_futures=CancellationPolicy.ALL) as executor:
        future = executor.submit(lambda t: time.sleep(t), 5)

    while future._state != "FINISHED":
        time.sleep(0.1)

    assert future._state == "FINISHED"

    with pytest.raises((TaskCancelledError, RayTaskError)):
        future.result()

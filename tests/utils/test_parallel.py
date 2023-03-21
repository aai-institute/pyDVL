import operator
import os
import time
from functools import partial, reduce

import numpy as np
import pytest

from pydvl.utils.parallel import MapReduceJob, init_parallel_backend
from pydvl.utils.parallel.backend import available_cpus, effective_n_jobs
from pydvl.utils.parallel.futures import init_executor
from pydvl.utils.parallel.map_reduce import _get_value


def test_effective_n_jobs(parallel_config, num_workers):
    parallel_backend = init_parallel_backend(parallel_config)
    if parallel_config.backend == "sequential":
        assert parallel_backend.effective_n_jobs(1) == 1
        assert parallel_backend.effective_n_jobs(4) == 1
        assert parallel_backend.effective_n_jobs(-1) == 1
    else:
        assert parallel_backend.effective_n_jobs(1) == 1
        assert parallel_backend.effective_n_jobs(4) == 4
        if parallel_config.address is None:
            assert parallel_backend.effective_n_jobs(-1) == available_cpus()
        else:
            assert parallel_backend.effective_n_jobs(-1) == num_workers

    for n_jobs in [-1, 1, 2]:
        assert parallel_backend.effective_n_jobs(n_jobs) == effective_n_jobs(
            n_jobs, parallel_config
        )
        assert effective_n_jobs(n_jobs, parallel_config) > 0

    with pytest.raises(ValueError):
        parallel_backend.effective_n_jobs(0)


@pytest.fixture()
def map_reduce_job_and_parameters(parallel_config, n_jobs, request):
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
            config=parallel_config,
            n_jobs=n_jobs,
        )
    elif kind == "list":
        map_reduce_job = partial(
            MapReduceJob,
            map_func=lambda x: x,
            reduce_func=lambda r: reduce(operator.add, r, []),
            config=parallel_config,
            n_jobs=n_jobs,
        )
    elif kind == "range":
        map_reduce_job = partial(
            MapReduceJob,
            map_func=lambda x: list(x),
            reduce_func=lambda r: reduce(operator.add, list(r), []),
            config=parallel_config,
            n_jobs=n_jobs,
        )
    elif kind == "custom":
        map_reduce_job = partial(
            MapReduceJob,
            map_func=map_func,
            reduce_func=reduce_func,
            config=parallel_config,
            n_jobs=n_jobs,
        )
    else:
        map_reduce_job = partial(
            MapReduceJob,
            map_func=lambda x: x * x,
            reduce_func=lambda r: r,
            config=parallel_config,
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
    if not isinstance(result, np.ndarray):
        assert result == expected
    else:
        assert (result == expected).all()


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
def test_chunkification(data, n_chunks, expected_chunks):
    map_reduce_job = MapReduceJob([], map_func=lambda x: x)
    chunks = list(map_reduce_job._chunkify(data, n_chunks))
    chunks = map_reduce_job.parallel_backend.get(chunks)
    for x, y in zip(chunks, expected_chunks):
        if not isinstance(x, np.ndarray):
            assert x == y
        else:
            assert (x == y).all()


@pytest.mark.parametrize(
    "max_parallel_tasks, n_finished, n_dispatched, expected_n_finished",
    [
        (1, 3, 6, 5),
        (3, 3, 3, 3),
        (10, 1, 15, 5),
        (20, 1, 3, 1),
    ],
)
def test_backpressure(
    max_parallel_tasks, n_finished, n_dispatched, expected_n_finished
):
    def map_func(x):
        import time

        time.sleep(1)
        return x

    inputs_ = list(range(n_dispatched))

    map_reduce_job = MapReduceJob(
        inputs_,
        map_func=map_func,
        max_parallel_tasks=max_parallel_tasks,
        timeout=10,
    )

    map_func = map_reduce_job._wrap_function(map_func)
    jobs = [map_func(x) for x in inputs_]
    n_finished = map_reduce_job._backpressure(
        jobs, n_finished=n_finished, n_dispatched=n_dispatched
    )
    assert n_finished == expected_n_finished


def test_map_reduce_job_partial_map_and_reduce_func(parallel_config):
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
        config=parallel_config,
    )
    result = map_reduce_job()
    assert result == 150


@pytest.mark.parametrize(
    "x, expected_x",
    [
        (None, None),
        ([0, 1], [0, 1]),
        (np.arange(3), np.arange(3)),
    ],
)
def test_map_reduce_get_value(x, expected_x, parallel_config):
    assert np.all(_get_value(x) == expected_x)
    parallel_backend = init_parallel_backend(parallel_config)
    x_id = parallel_backend.put(x)
    assert np.all(_get_value(x_id) == expected_x)


def test_wrap_function(parallel_config, num_workers):
    def fun(x, **kwargs):
        return dict(x=x * x, **kwargs)

    parallel_backend = init_parallel_backend(parallel_config)
    # Try two kwargs for @ray.remote. Should be ignored in the sequential backend
    wrapped_func = parallel_backend.wrap(fun, num_cpus=1, max_calls=1)
    x = parallel_backend.put(2)
    ret = parallel_backend.get(wrapped_func(x))

    assert ret["x"] == 4
    assert len(ret) == 1  # Ensure that kwargs are not passed to the function

    if parallel_config.backend != "sequential":
        # Test that the function is executed in different processes
        def get_pid():
            time.sleep(2)  # FIXME: waiting less means fewer processes are used?!
            return os.getpid()

        wrapped_func = parallel_backend.wrap(get_pid, num_cpus=1)
        pids = parallel_backend.get([wrapped_func() for _ in range(num_workers)])
        assert len(set(pids)) == num_workers


def test_futures_executor_submit(parallel_config):
    with init_executor(config=parallel_config) as executor:
        future = executor.submit(lambda x: x + 1, 1)
        result = future.result()
    assert result == 2


def test_futures_executor_map(parallel_config):
    with init_executor(config=parallel_config) as executor:
        results = list(executor.map(lambda x: x + 1, range(3)))
    assert results == [1, 2, 3]


def test_futures_executor_map_with_max_workers(parallel_config, num_workers):
    if parallel_config.backend != "ray":
        pytest.skip("Currently this test only works with Ray")

    def func(_):
        time.sleep(1)
        return time.monotonic()

    start_time = time.monotonic()
    with init_executor(config=parallel_config) as executor:
        assert executor._max_workers == num_workers
        list(executor.map(func, range(3)))
    end_time = time.monotonic()
    total_time = end_time - start_time
    # We expect the time difference to be > 3 / num_workers, but has to be at least 1
    assert total_time > max(1.0, 3 / num_workers)

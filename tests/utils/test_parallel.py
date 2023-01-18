import operator
from functools import partial, reduce

import numpy as np
import pytest

from pydvl.utils.parallel import MapReduceJob, init_parallel_backend
from pydvl.utils.parallel.backend import available_cpus
from pydvl.utils.parallel.map_reduce import _get_value


def test_effective_n_jobs(parallel_config):
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
            assert parallel_backend.effective_n_jobs(-1) == 4


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

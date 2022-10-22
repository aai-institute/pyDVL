import operator
from functools import reduce
from itertools import zip_longest

import numpy as np
import pytest

from pydvl.utils.parallel import MapReduceJob


@pytest.fixture()
def map_reduce_job(request):
    try:
        kind, map_func, reduce_func = request.param
        assert kind == "custom"
    except ValueError:
        kind = request.param
    if kind == "numpy":
        return MapReduceJob(map_func=np.sum, reduce_func=np.sum)
    elif kind == "list":
        return MapReduceJob(
            map_func=lambda x: x, reduce_func=lambda r: reduce(operator.add, r, [])
        )
    elif kind == "range":
        return MapReduceJob(
            map_func=lambda x: list(x),
            reduce_func=lambda r: reduce(operator.add, list(r), []),
        )
    elif kind == "custom":
        return MapReduceJob(
            map_func=map_func,
            reduce_func=reduce_func,
        )
    else:
        return MapReduceJob(map_func=lambda x: x * x, reduce_func=lambda r: r)


@pytest.mark.parametrize(
    "map_reduce_job, indices, expected",
    [
        ("list", [], [[]]),
        ("list", [1, 2], [[1, 2]]),
        ("list", [1, 2, 3], [[1, 2, 3]]),
        ("range", range(10), [list(range(10))]),
        ("numpy", list(range(10)), [45]),
    ],
    indirect=["map_reduce_job"],
)
@pytest.mark.parametrize("n_jobs", [1])
@pytest.mark.parametrize("n_runs", [1, 2])
def test_map_reduce_job(map_reduce_job, indices, n_jobs, n_runs, expected):
    result = map_reduce_job(
        indices,
        n_jobs=n_jobs,
        n_runs=n_runs,
    )
    for exp, ret in zip_longest(expected * n_runs, result, fillvalue=None):
        if not isinstance(ret, np.ndarray):
            assert ret == exp
        else:
            assert (ret == exp).all()


@pytest.mark.parametrize(
    "map_reduce_job, indices, expected",
    [
        ("list", [], [[]]),
        ("list", [1, 2], [[1, 2]]),
        ("list", [1, 2, 3, 4], [[1, 2, 3, 4]]),
        ("range", range(10), [list(range(10))]),
        ("numpy", list(range(10)), [45]),
    ],
    indirect=["map_reduce_job"],
)
@pytest.mark.parametrize("n_jobs", [2, 4])
@pytest.mark.parametrize("n_runs", [1, 2])
def test_map_reduce_job_chunkified_inputs(
    map_reduce_job, indices, n_jobs, n_runs, expected
):
    result = map_reduce_job(indices, n_jobs=n_jobs, n_runs=n_runs, chunkify_inputs=True)
    assert len(result) == n_runs
    for exp, ret in zip_longest(expected * n_runs, result, fillvalue=None):
        if not isinstance(ret, np.ndarray):
            assert ret == exp
        else:
            assert (ret == exp).all()


@pytest.mark.parametrize(
    "data, n_chunks, expected_chunks",
    [
        ([], 3, []),
        ([1, 2, 3], 2, [[1, 2], [3]]),
        ([1, 2, 3, 4], 2, [[1, 2], [3, 4]]),
        ([1, 2, 3, 4], 3, [[1, 2], [3], [4]]),
        ([1, 2, 3, 4], 5, [[1], [2], [3], [4]]),
        (range(10), 4, [range(0, 3), range(3, 6), range(6, 8), range(8, 10)]),
    ],
)
def test_chunkification(data, n_chunks, expected_chunks):
    chunks = list(MapReduceJob._chunkify(data, n_chunks))
    assert chunks == expected_chunks


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

    map_reduce_job = MapReduceJob(
        map_func=map_func,
        max_parallel_tasks=max_parallel_tasks,
        timeout=10,
    )
    map_func = map_reduce_job._wrap_function(map_func)
    jobs = [map_func(x) for x in range(n_dispatched)]
    n_finished = map_reduce_job._backpressure(
        jobs, n_finished=n_finished, n_dispatched=n_dispatched
    )
    assert n_finished == expected_n_finished


# TODO: figure out test cases for this test
@pytest.mark.skip
@pytest.mark.parametrize(
    "map_reduce_job, indices, n_jobs, n_runs, expected",
    [
        ("other", [], 1, 1, [[]]),
    ],
    indirect=["map_reduce_job"],
)
def test_map_reduce_job_expected_failures(
    map_reduce_job, indices, n_jobs, n_runs, expected
):
    with pytest.raises(expected):
        map_reduce_job(indices, n_jobs=n_jobs, n_runs=n_runs)

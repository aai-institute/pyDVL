import operator
import pytest
import numpy as np

from functools import reduce
from valuation.utils.parallel import MapReduceJob, map_reduce

np_fun = MapReduceJob.from_fun(np.sum, np.sum)
list_fun = MapReduceJob.from_fun(lambda x: x,
                                 lambda r: reduce(operator.add, r, []))
range_fun = MapReduceJob.from_fun(lambda x: list(x),
                                  lambda r: reduce(operator.add, list(r), []))
# dict_fun = MapReduceJob.from_fun(
#         lambda x: x, lambda r: reduce(lambda x, y: dict(x, **y), r, {}))

run_parallel_test_data = []
# 'multiprocessing' not supported because pickle cannot serialize lambdas
# Could be fixed, but is it worth it?
for backend in ('loky', 'sequential', 'threading'):
    run_parallel_test_data.extend(
            [(list_fun, [], 1, 1, backend, [[]]),
             (range_fun, range(10), 1, 1, backend, [list(range(10))]),
             (range_fun, range(10), 4, 2, backend, [list(range(10))]),
             (np_fun, np.arange(10), 1, 1, backend, [45]),
             (np_fun, np.arange(10), 2, 2, backend, [45] * 2),
             (np_fun, np.arange(10), 4, 2, backend, [45] * 2),
             (np_fun, np.arange(10), 2, 4, backend, [45] * 4),
             (np_fun, np.arange(10), 10, 4, backend, [45] * 4),
             (np_fun, np.arange(10), 3, 7, backend, [45] * 7),
             (np_fun, np.arange(10), 12, 1, backend, [45]),
             ])


@pytest.mark.parametrize(
    "fun, indices, num_jobs, num_runs, backend, expected", run_parallel_test_data)
def test_run_parallel(fun, indices, num_jobs, num_runs, backend, expected):

    if isinstance(expected, Exception):
        with pytest.raises(expected):
            map_reduce(fun, indices,
                       num_jobs=num_jobs, num_runs=num_runs, backend=backend)
    else:
        result = map_reduce(fun, indices,
                            num_jobs=num_jobs, num_runs=num_runs, backend=backend)
        for exp, ret in zip(expected, result):
            if not isinstance(ret, np.ndarray):
                assert ret == exp
            else:
                assert (ret == exp).all()


@pytest.mark.skip("To do")
def test_chunkify():
    # TODO: test generation of job and run ids
    pass

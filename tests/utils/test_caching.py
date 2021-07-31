import numpy as np

from time import sleep
from typing import Iterable
from valuation.utils import memcached, map_reduce, MapReduceJob
from valuation.utils.logging import start_logging_server, _logger


def test_memcached_single_job(memcached_client):
    # TODO: maybe this should be a fixture too...
    @memcached(server=memcached_client.server,
               threshold=0)  # Always cache results
    def foo(indices: Iterable[int]) -> float:
        return float(np.sum(indices))

    n = 1000

    foo(np.arange(n))
    hits_before = memcached_client.stats()[b'get_hits']
    foo(np.arange(n))
    hits_after = memcached_client.stats()[b'get_hits']

    assert hits_after > hits_before


def test_memcached_parallel_jobs(memcached_client):

    server = start_logging_server()

    # Note that we typically do NOT want to ignore run_id
    @memcached(server=memcached_client.server,
               threshold=0,  # Always cache results
               ignore_args=['job_id', 'run_id'])
    def foo(indices: Iterable[int], job_id: int, run_id: int) -> float:
        _logger.info(f"run_id: {run_id}, waiting...")
        sleep(0.5 * run_id)
        _logger.info(f"run_id: {run_id}, running")
        return float(np.sum(indices))

    # log_server = start_logging_server()

    n = 1234
    nruns = 4
    job = MapReduceJob.from_fun(foo, np.sum)

    hits_before = memcached_client.stats()[b'get_hits']
    result = map_reduce(job, data=np.arange(n),
                        num_jobs=1, num_runs=nruns)
    hits_after = memcached_client.stats()[b'get_hits']

    assert result[0] == n*(n-1)/2  # Sanity check
    assert hits_after - hits_before >= nruns - 1  # meh...
    _logger.info(f'before: {hits_before}, after: {hits_after}')

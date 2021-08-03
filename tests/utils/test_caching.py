import numpy as np

from typing import Iterable
from valuation.utils import memcached, map_reduce, MapReduceJob


def test_memcached_single_job(memcached_client):
    client, config = memcached_client

    # TODO: maybe this should be a fixture too...
    @memcached(client_config=config, threshold=0)  # Always cache results
    def foo(indices: Iterable[int]) -> float:
        return float(np.sum(indices))

    n = 1000
    foo(np.arange(n))
    hits_before = client.stats()[b'get_hits']
    foo(np.arange(n))
    hits_after = client.stats()[b'get_hits']

    assert hits_after > hits_before


def test_memcached_parallel_jobs(memcached_client):
    client, config = memcached_client

    @memcached(client_config=config,
               threshold=0,  # Always cache results
               # Note that we typically do NOT want to ignore run_id
               ignore_args=['job_id', 'run_id'])
    def foo(indices: Iterable[int], job_id: int, run_id: int) -> float:
        # from valuation.utils.logging import logger
        # logger.info(f"run_id: {run_id}, running...")
        return float(np.sum(indices))

    n = 1234
    num_runs = 10
    hits_before = client.stats()[b'get_hits']
    job = MapReduceJob.from_fun(foo, np.sum)
    result = map_reduce(job, data=np.arange(n), num_jobs=4, num_runs=num_runs)
    hits_after = client.stats()[b'get_hits']

    assert result[0] == n*(n-1)/2  # Sanity check
    # FIXME! This is non-deterministic: if packets are delayed for longer than
    #  the timeout configured then we won't have nruns hits. So we add this
    #  good old hard-coded magic number here.
    assert hits_after - hits_before >= nruns - 2

import logging
from time import sleep, time

import numpy as np
import pytest
from numpy.typing import NDArray

from pydvl.utils import MapReduceJob, memcached

logger = logging.getLogger(__name__)


def test_failed_connection():
    from pydvl.utils import MemcachedClientConfig

    client_config = MemcachedClientConfig(server=("localhost", 0), connect_timeout=0.1)
    with pytest.raises((ConnectionRefusedError, OSError)):
        memcached(client_config)(lambda x: x)


def test_memcached_single_job(memcached_client):
    client, config = memcached_client

    # TODO: maybe this should be a fixture too...
    @memcached(client_config=config, time_threshold=0)  # Always cache results
    def foo(indices: NDArray[np.int_]) -> float:
        return float(np.sum(indices))

    n = 1000
    foo(np.arange(n))
    hits_before = client.stats()[b"get_hits"]
    foo(np.arange(n))
    hits_after = client.stats()[b"get_hits"]

    assert hits_after > hits_before


def test_memcached_parallel_jobs(memcached_client, parallel_config):
    client, config = memcached_client

    @memcached(
        client_config=config,
        time_threshold=0,  # Always cache results
        # Note that we typically do NOT want to ignore run_id
        ignore_args=["job_id", "run_id"],
    )
    def foo(indices: NDArray[np.int_], *args, **kwargs) -> float:
        # logger.info(f"run_id: {run_id}, running...")
        return float(np.sum(indices))

    n = 1234
    n_runs = 10
    hits_before = client.stats()[b"get_hits"]

    map_reduce_job = MapReduceJob(
        np.arange(n), foo, np.sum, n_jobs=4, config=parallel_config
    )
    results = []

    for _ in range(n_runs):
        result = map_reduce_job()
        results.append(result)
    hits_after = client.stats()[b"get_hits"]

    assert results[0] == n * (n - 1) / 2  # Sanity check
    # FIXME! This is non-deterministic: if packets are delayed for longer than
    #  the timeout configured then we won't have num_runs hits. So we add this
    #  good old hard-coded magic number here.
    assert hits_after - hits_before >= n_runs - 2


def test_memcached_repeated_training(memcached_client):
    _, config = memcached_client

    @memcached(
        client_config=config,
        time_threshold=0,  # Always cache results
        allow_repeated_evaluations=True,
        rtol_stderr=0.01,
        # Note that we typically do NOT want to ignore run_id
        ignore_args=["job_id", "run_id"],
    )
    def foo(indices: NDArray[np.int_]) -> float:
        # from pydvl.utils.logging import logger
        # logger.info(f"run_id: {run_id}, running...")
        return float(np.sum(indices)) + np.random.normal(scale=10)

    n = 7
    foo(np.arange(n))
    for _ in range(10_000):
        result = foo(np.arange(n))

    assert (result - np.sum(np.arange(n))) < 1
    assert foo.stats.sets < foo.stats.hits


def test_memcached_faster_with_repeated_training(memcached_client):
    _, config = memcached_client

    @memcached(
        client_config=config,
        time_threshold=0,  # Always cache results
        allow_repeated_evaluations=True,
        rtol_stderr=0.1,
        # Note that we typically do NOT want to ignore run_id
        ignore_args=["job_id", "run_id"],
    )
    def foo_cache(indices: NDArray[np.int_]) -> float:
        # from pydvl.utils.logging import logger
        # logger.info(f"run_id: {run_id}, running...")
        sleep(0.01)
        return float(np.sum(indices)) + np.random.normal(scale=1)

    def foo_no_cache(indices: NDArray[np.int_]) -> float:
        # from pydvl.utils.logging import logger
        # logger.info(f"run_id: {run_id}, running...")
        sleep(0.01)
        return float(np.sum(indices)) + np.random.normal(scale=1)

    n = 3
    foo_cache(np.arange(n))
    foo_no_cache(np.arange(n))

    start = time()
    for _ in range(300):
        result_fast = foo_cache(np.arange(n))
    end = time()
    fast_time = end - start

    start = time()
    results_slow = []
    for _ in range(300):
        result = foo_no_cache(np.arange(n))
        results_slow.append(result)
    end = time()
    slow_time = end - start

    assert (result_fast - np.mean(results_slow)) < 1
    assert fast_time < slow_time


@pytest.mark.parametrize("n, atol", [(10, 4), (20, 10)])
@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("n_runs", [100])
def test_memcached_parallel_repeated_training(
    memcached_client, n, atol, n_jobs, n_runs, parallel_config, seed=42
):
    _, config = memcached_client
    np.random.seed(seed)

    @memcached(
        client_config=config,
        time_threshold=0,  # Always cache results
        allow_repeated_evaluations=True,
        rtol_stderr=0.01,
        # Note that we typically do NOT want to ignore run_id
        ignore_args=["job_id", "run_id"],
    )
    def map_func(indices: NDArray[np.int_]) -> float:
        # from pydvl.utils.logging import logger
        # logger.info(f"run_id: {run_id}, running...")
        return np.sum(indices).item() + np.random.normal(scale=5)

    def reduce_func(chunks: NDArray[np.float_]) -> float:
        return np.sum(chunks).item()

    map_reduce_job = MapReduceJob(
        np.arange(n), map_func, reduce_func, n_jobs=n_jobs, config=parallel_config
    )
    results = []
    for _ in range(n_runs):
        result = map_reduce_job()
        results.append(result)

    exact_value = np.sum(np.arange(n)).item()

    assert np.isclose(results[-1], results[-2], atol=atol)
    assert np.isclose(results[-1], exact_value, atol=atol)

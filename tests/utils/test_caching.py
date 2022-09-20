import logging
import os
from time import sleep, time
from typing import Iterable

import numpy as np
import pytest

from valuation.utils import MapReduceJob, map_reduce, memcached
from valuation.utils.caching import get_running_avg_variance

log = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "numbers_series",
    [
        ([3, 4, 5, 6]),
        (list(range(10))),
        (np.linspace(1, 4, 10)),
    ],
)
def test_get_running_avg_variance(numbers_series):
    true_avg = np.mean(numbers_series)
    true_var = np.var(numbers_series)

    prev_avg = np.mean(numbers_series[:-1])
    prev_var = np.var(numbers_series[:-1])
    new_value = numbers_series[-1]
    count = len(numbers_series) - 1
    new_avg, new_var = get_running_avg_variance(prev_avg, prev_var, new_value, count)
    assert new_avg == true_avg
    assert new_var == true_var


def test_memcached_single_job(memcached_client):
    client, config = memcached_client

    # TODO: maybe this should be a fixture too...
    @memcached(client_config=config, cache_threshold=0)  # Always cache results
    def foo(indices: Iterable[int]) -> float:
        return float(np.sum(indices))

    n = 1000
    foo(np.arange(n))
    hits_before = client.stats()[b"get_hits"]
    foo(np.arange(n))
    hits_after = client.stats()[b"get_hits"]

    assert hits_after > hits_before


def test_memcached_parallel_jobs(memcached_client):
    client, config = memcached_client

    @memcached(
        client_config=config,
        cache_threshold=0,  # Always cache results
        # Note that we typically do NOT want to ignore run_id
        ignore_args=["job_id", "run_id"],
    )
    def foo(indices: Iterable[int], *args, **kwargs) -> float:
        # from valuation.utils.logging import logger
        # logger.info(f"run_id: {run_id}, running...")
        return float(np.sum(indices))

    n = 1234
    n_runs = 10
    hits_before = client.stats()[b"get_hits"]
    map_reduce_job = MapReduceJob(foo, np.sum, n_jobs=4, n_runs=n_runs)
    result = map_reduce_job(np.arange(n))
    hits_after = client.stats()[b"get_hits"]

    assert result[0] == n * (n - 1) / 2  # Sanity check
    # FIXME! This is non-deterministic: if packets are delayed for longer than
    #  the timeout configured then we won't have num_runs hits. So we add this
    #  good old hard-coded magic number here.
    assert hits_after - hits_before >= n_runs - 2


def test_memcached_repeated_training(memcached_client):
    _, config = memcached_client

    @memcached(
        client_config=config,
        cache_threshold=0,  # Always cache results
        # Note that we typically do NOT want to ignore run_id
        allow_repeated_training=True,
        rtol_threshold=0.01,
        ignore_args=["job_id", "run_id"],
    )
    def foo(indices: Iterable[int]) -> float:
        # from valuation.utils.logging import logger
        # logger.info(f"run_id: {run_id}, running...")
        return float(np.sum(indices)) + np.random.normal(scale=10)

    n = 7
    foo(np.arange(n))
    for _ in range(10_000):
        result = foo(np.arange(n))

    assert (result - np.sum(np.arange(n))) < 1
    assert foo.cache_info.sets < foo.cache_info.hits


def test_memcached_faster_with_repeated_training(memcached_client):
    _, config = memcached_client

    @memcached(
        client_config=config,
        cache_threshold=0,  # Always cache results
        # Note that we typically do NOT want to ignore run_id
        allow_repeated_training=True,
        rtol_threshold=0.1,
        ignore_args=["job_id", "run_id"],
    )
    def foo_cache(indices: Iterable[int]) -> float:
        # from valuation.utils.logging import logger
        # logger.info(f"run_id: {run_id}, running...")
        sleep(0.01)
        return float(np.sum(indices)) + np.random.normal(scale=1)

    def foo_no_cache(indices: Iterable[int]) -> float:
        # from valuation.utils.logging import logger
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


@pytest.mark.parametrize(
    "n, atol",
    [
        (7, 3),
        (10, 3),
        (20, 10),
    ],
)
def test_memcached_parallel_repeated_training(memcached_client, n, atol, seed=42):
    _, config = memcached_client
    np.random.seed(seed)

    @memcached(
        client_config=config,
        cache_threshold=0,  # Always cache results
        # Note that we typically do NOT want to ignore run_id
        allow_repeated_training=True,
        rtol_threshold=0.01,
        ignore_args=["job_id", "run_id"],
    )
    def map_func(indices: Iterable[int]) -> float:
        # from valuation.utils.logging import logger
        # logger.info(f"run_id: {run_id}, running...")
        return np.sum(indices).item() + np.random.normal(scale=10)

    def reduce_func(chunks: Iterable[float]) -> float:
        return np.sum(chunks).item()

    n_runs = 100
    map_reduce_job = MapReduceJob(map_func, reduce_func, n_jobs=2, n_runs=n_runs)
    result = map_reduce_job(np.arange(n))

    exact_value = np.sum(np.arange(n)).item()

    assert np.isclose(result[-1], result[-2], atol=atol)
    assert np.isclose(result[-1], exact_value, atol=atol)

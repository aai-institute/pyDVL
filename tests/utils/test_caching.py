import logging
import pickle
import tempfile
from time import sleep, time
from typing import Optional

import numpy as np
import pytest
from numpy.typing import NDArray

from pydvl.parallel import MapReduceJob
from pydvl.parallel.config import ParallelConfig
from pydvl.utils.caching import (
    CachedFunc,
    CachedFuncConfig,
    DiskCacheBackend,
    InMemoryCacheBackend,
    MemcachedCacheBackend,
)
from pydvl.utils.types import Seed

from ..conftest import num_workers

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def parallel_config():
    return ParallelConfig(backend="joblib", n_cpus_local=num_workers())


def foo(indices: NDArray[np.int_], *args, **kwargs) -> float:
    return float(np.sum(indices))


def foo_duplicate(indices: NDArray[np.int_], *args, **kwargs) -> float:
    return float(np.sum(indices))


def foo_with_random(indices: NDArray[np.int_], *args, **kwargs) -> float:
    rng = np.random.default_rng()
    scale = kwargs.get("scale", 0.5)
    return float(np.sum(indices) + rng.normal(scale=scale))


def foo_with_random_and_sleep(indices: NDArray[np.int_], *args, **kwargs) -> float:
    sleep(0.01)
    rng = np.random.default_rng()
    scale = kwargs.get("scale", 0.5)
    return float(np.sum(indices)) + rng.normal(scale=scale)


# Used to test caching of methods
class CacheTest:
    def __init__(self):
        self.value = 0

    def foo(self):
        return 1


@pytest.fixture(params=["in-memory", "disk", "memcached"])
def cache_backend(request):
    backend: str = request.param
    if backend == "in-memory":
        cache_backend = InMemoryCacheBackend()
        yield cache_backend
        cache_backend.clear()
    elif backend == "disk":
        with tempfile.TemporaryDirectory() as tempdir:
            cache_backend = DiskCacheBackend(tempdir)
            yield cache_backend
            cache_backend.clear()
    elif backend == "memcached":
        try:
            cache_backend = MemcachedCacheBackend()
        except ConnectionRefusedError as e:
            raise RuntimeError(
                f"Could not connected to Memcached server. original error message: {str(e)}"
            )
        yield cache_backend
        cache_backend.clear()
    else:
        raise ValueError(f"Unknown cache backend {backend}")


@pytest.mark.parametrize(
    "f1, f2, expected_equal",
    [
        # Test that the same function gets the same hash
        (lambda x: x, lambda x: x, True),
        (foo, foo, True),
        # Test that different functions get different hashes
        (foo, lambda x: x, False),
        (foo, foo_with_random, False),
        (foo_with_random, foo_with_random_and_sleep, False),
        # Test that functions with different names but the same body get different hashes
        (foo, foo_duplicate, True),
    ],
)
def test_cached_func_hash_function(f1, f2, expected_equal):
    f1_hash = CachedFunc._hash_function(f1)
    f2_hash = CachedFunc._hash_function(f2)
    if expected_equal:
        assert f1_hash == f2_hash, f"{f1_hash} != {f2_hash}"
    else:
        assert f1_hash != f2_hash, f"{f1_hash} == {f2_hash}"


@pytest.mark.parametrize(
    "args1, args2, expected_equal",
    [
        # Test that the same arguments get the same hash
        ([[]], [[]], True),
        ([[1]], [[1]], True),
        ([frozenset([])], [frozenset([])], True),
        ([frozenset([1])], [frozenset([1])], True),
        ([np.ones(3)], [np.ones(3)], True),
        ([np.ones(3), 16], [np.ones(3), 16], True),
        ([frozenset(np.ones(3))], [frozenset(np.ones(3))], True),
        # Test that different arguments get different hashes
        ([[1, 2, 3]], [np.ones(3)], False),
        ([np.ones(3)], [np.ones(5)], False),
        ([np.ones(3)], [frozenset(np.ones(3))], False),
    ],
)
def test_cached_func_hash_arguments(args1, args2, expected_equal):
    args1_hash = CachedFunc._hash_arguments(foo, ignore_args=[], args=args1, kwargs={})
    args2_hash = CachedFunc._hash_arguments(foo, ignore_args=[], args=args2, kwargs={})
    if expected_equal:
        assert args1_hash == args2_hash, f"{args1_hash} != {args2_hash}"
    else:
        assert args1_hash != args2_hash, f"{args1_hash} == {args2_hash}"


def test_cached_func_hash_arguments_of_method():
    obj = CacheTest()

    hash1 = CachedFunc._hash_arguments(obj.foo, [], tuple(), {})
    obj.value += 1
    hash2 = CachedFunc._hash_arguments(obj.foo, [], tuple(), {})
    assert hash1 == hash2


def test_cache_backend_serialization(cache_backend):
    value = 16.8
    cache_backend.set("key", value)
    deserialized_cache_backend = pickle.loads(pickle.dumps(cache_backend))
    assert deserialized_cache_backend.get("key") == value
    if isinstance(cache_backend, InMemoryCacheBackend):
        assert cache_backend.cached_values == deserialized_cache_backend.cached_values
    elif isinstance(cache_backend, DiskCacheBackend):
        assert cache_backend.cache_dir == deserialized_cache_backend.cache_dir


def test_single_job(cache_backend):
    cached_func_config = CachedFuncConfig(time_threshold=0.0)
    wrapped_foo = cache_backend.wrap(foo, config=cached_func_config)

    n = 1000
    wrapped_foo(np.arange(n))
    hits_before = wrapped_foo.stats.hits
    wrapped_foo(np.arange(n))
    hits_after = wrapped_foo.stats.hits

    assert hits_after > hits_before


def test_without_pymemcache(mocker):
    import importlib
    import sys

    mocker.patch.dict("sys.modules", {"pymemcache": None})
    with pytest.raises(ModuleNotFoundError) as err:
        importlib.reload(sys.modules["pydvl.utils.caching.memcached"])

    # error message should contain the extra install expression
    assert "pyDVL[memcached]" in err.value.msg


def test_memcached_failed_connection():
    from pydvl.utils import MemcachedClientConfig

    config = MemcachedClientConfig(server=("localhost", 0), connect_timeout=0.1)
    with pytest.raises((ConnectionRefusedError, OSError)):
        MemcachedCacheBackend(config)


def test_cache_time_threshold(cache_backend):
    cached_func_config = CachedFuncConfig(time_threshold=1.0)
    wrapped_foo = cache_backend.wrap(foo, config=cached_func_config)

    n = 1000
    indices = np.arange(n)
    wrapped_foo(indices)
    hits_before = wrapped_foo.stats.hits
    misses_before = wrapped_foo.stats.misses
    wrapped_foo(indices)
    hits_after = wrapped_foo.stats.hits
    misses_after = wrapped_foo.stats.misses

    assert hits_after == hits_before
    assert misses_after > misses_before


def test_cache_ignore_args(cache_backend):
    # Note that we typically do NOT want to ignore run_id
    cached_func_config = CachedFuncConfig(
        time_threshold=0.0,
        ignore_args=["job_id"],
    )
    wrapped_foo = cache_backend.wrap(foo, config=cached_func_config)

    n = 1000
    indices = np.arange(n)
    wrapped_foo(indices, job_id=0)
    hits_before = wrapped_foo.stats.hits
    wrapped_foo(indices, job_id=16)
    hits_after = wrapped_foo.stats.hits

    assert hits_after > hits_before


def test_parallel_jobs(cache_backend, parallel_config):
    if not isinstance(cache_backend, MemcachedCacheBackend):
        pytest.skip("Only running this test with MemcachedCacheBackend")

    # Note that we typically do NOT want to ignore run_id
    cached_func_config = CachedFuncConfig(
        ignore_args=["job_id", "run_id"], time_threshold=0
    )
    wrapped_foo = cache_backend.wrap(foo, config=cached_func_config)

    n = 1234
    n_runs = 10
    hits_before = cache_backend.client.stats()[b"get_hits"]

    map_reduce_job = MapReduceJob(
        np.arange(n), wrapped_foo, np.sum, n_jobs=4, config=parallel_config
    )
    results = []

    for _ in range(n_runs):
        result = map_reduce_job()
        results.append(result)
    hits_after = cache_backend.client.stats()[b"get_hits"]

    assert results[0] == n * (n - 1) / 2  # Sanity check
    # FIXME! This is non-deterministic: if packets are delayed for longer than
    #  the timeout configured then we won't have num_runs hits. So we add this
    #  good old hard-coded magic number here.
    assert hits_after - hits_before >= n_runs - 2, wrapped_foo.stats


def test_repeated_training(cache_backend, worker_id: str):
    cached_func_config = CachedFuncConfig(
        time_threshold=0.0,
        allow_repeated_evaluations=True,
        rtol_stderr=0.01,
    )
    wrapped_foo = cache_backend.wrap(
        foo_with_random,
        config=cached_func_config,
    )

    n = 7
    indices = np.arange(n)

    for _ in range(1_000):
        result = wrapped_foo(indices, worker_id)

    np.testing.assert_allclose(result, np.sum(indices), atol=1)
    assert wrapped_foo.stats.sets < wrapped_foo.stats.hits


def test_faster_with_repeated_training(cache_backend, worker_id: str):
    cached_func_config = CachedFuncConfig(
        time_threshold=0.0,
        allow_repeated_evaluations=True,
        rtol_stderr=0.1,
    )
    wrapped_foo = cache_backend.wrap(
        foo_with_random_and_sleep,
        config=cached_func_config,
    )

    n = 3
    n_repetitions = 500
    indices = np.arange(n)

    start = time()
    for _ in range(n_repetitions):
        result_fast = wrapped_foo(indices, worker_id)
    end = time()
    fast_time = end - start

    start = time()
    results_slow = []
    for _ in range(n_repetitions):
        result = foo_with_random_and_sleep(indices, worker_id)
        results_slow.append(result)
    end = time()
    slow_time = end - start

    np.testing.assert_allclose(np.mean(results_slow), np.sum(indices), atol=0.1)
    np.testing.assert_allclose(result_fast, np.mean(results_slow), atol=1)
    assert fast_time < slow_time


@pytest.mark.parametrize("n, atol", [(10, 5), (20, 10)])
@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("n_runs", [20])
def test_parallel_repeated_training(
    cache_backend, n, atol, n_jobs, n_runs, parallel_config
):
    def map_func(indices: NDArray[np.int_], seed: Optional[Seed] = None) -> float:
        return np.sum(indices).item() + np.random.normal(scale=1)

    # Note that we typically do NOT want to ignore run_id
    cached_func_config = CachedFuncConfig(
        time_threshold=0.0,
        allow_repeated_evaluations=True,
        rtol_stderr=0.01,
        ignore_args=["job_id", "run_id"],
    )
    wrapped_map_func = cache_backend.wrap(
        map_func,
        config=cached_func_config,
    )

    def reduce_func(chunks: NDArray[np.float64]) -> float:
        return np.sum(chunks).item()

    map_reduce_job = MapReduceJob(
        np.arange(n),
        wrapped_map_func,
        reduce_func,
        n_jobs=n_jobs,
        config=parallel_config,
    )
    results = []
    for _ in range(n_runs):
        result = map_reduce_job()
        results.append(result)

    exact_value = np.sum(np.arange(n)).item()

    np.testing.assert_allclose(results[-1], results[-2], atol=atol)
    np.testing.assert_allclose(results[-1], exact_value, atol=atol)

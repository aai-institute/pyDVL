import logging
import os
import platform
from dataclasses import asdict
from typing import Optional, Tuple

import numpy as np
import pytest
from pytest import Config, FixtureRequest
from sklearn.utils import Bunch

from pydvl.parallel import available_cpus
from pydvl.utils import Dataset
from tests.cache import CloudPickleCache


def pytest_addoption(parser):
    parser.addoption(
        "--memcached-service",
        action="store_true",
        default="localhost:11211",
        help="Address of memcached server to use for tests.",
    )
    parser.addoption(
        "--slow-tests",
        action="store_true",
        help="Run tests marked as slow using the @slow marker",
    )
    parser.addoption(
        "--with-cuda",
        action="store_true",
        default=False,
        help="Set device fixture to 'cuda' if available",
    )


@pytest.fixture
def cache(request: "FixtureRequest") -> CloudPickleCache:
    """Return a cache object that can persist state between testing sessions.

    ```pycon
    cache.get(key, default)
    cache.set(key, value)
    ```

    Keys must be ``/`` separated strings, where the first part is usually the
    name of your plugin or application to avoid clashes with other cache users.

    Values can be any object handled by the json stdlib module.
    """
    assert request.config.cloud_pickle_cache is not None
    return request.config.cloud_pickle_cache


@pytest.fixture()
def seed(request):
    try:
        return request.param
    except AttributeError:
        return 24


@pytest.fixture()
def seed_alt(request):
    try:
        return request.param
    except AttributeError:
        return 42


@pytest.fixture()
def collision_tol(request):
    return 0.01


@pytest.fixture(autouse=True)
def pytorch_seed(seed):
    try:
        import torch

        torch.manual_seed(seed)
        # TODO if necessary extract this into a separate fixture
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def is_memcache_responsive(hostname, port):
    from pymemcache.client import Client

    try:
        client = Client(server=(hostname, port))
        client.flush_all()
        return True
    except ConnectionRefusedError:
        return False


@pytest.fixture(scope="session")
def memcached_service(request) -> Tuple[str, int]:
    opt = request.config.getoption("--memcached-service", default="localhost:11211")
    host, port = opt.split(":")
    return host, int(port)


@pytest.fixture(scope="function")
def memcache_client_config(memcached_service) -> "MemcachedClientConfig":  # noqa: F821
    from pydvl.utils import MemcachedClientConfig

    return MemcachedClientConfig(
        server=memcached_service, connect_timeout=1.0, timeout=1, no_delay=True
    )


@pytest.fixture(scope="function")
def memcached_client(
    memcache_client_config,
) -> Tuple["Client", "MemcachedClientConfig"]:  # noqa: F821
    from pymemcache.client import Client

    try:
        c = Client(**asdict(memcache_client_config))
        c.flush_all()
        return c, memcache_client_config
    except Exception as e:
        raise ConnectionError(
            f"Could not connect to memcached at {memcache_client_config.server}"
        ) from e


@pytest.fixture(scope="function")
def linear_dataset(a: float, b: float, num_points: int):
    """Constructs a dataset sampling from y=ax+b + eps, with eps~Gaussian and
    x in [-1,1]

    Args:
        a: Slope
        b: intercept
        num_points: number of (x,y) samples to construct
        train_size: fraction of points to use for training (between 0 and 1)

    Returns:
        Dataset with train/test split. call str() on it to see the parameters
    """
    step = 2 / num_points
    stddev = 0.1
    x = np.arange(-1, 1, step)
    y = np.random.normal(loc=a * x + b, scale=stddev)
    db = Bunch()
    db.data, db.target = x.reshape(-1, 1), y
    db.DESCR = f"{{y_i~N({a}*x_i + {b}, {stddev:0.2f}): i=1, ..., {num_points}}}"
    db.feature_names = ["x"]
    db.target_names = ["y"]
    return Dataset.from_sklearn(data=db, train_size=0.3)


@pytest.fixture(autouse=True)
def seed_numpy(seed=42):
    np.random.seed(seed)


def num_workers() -> int:
    # Run with 2 CPUs inside GitHub actions
    if os.getenv("CI"):
        return 2
    # And a maximum of 4 CPUs locally (most tests don't really benefit from more)
    return max(1, min(available_cpus() - 1, 4))


@pytest.fixture(scope="session")
def n_jobs() -> int:
    return num_workers()


def pytest_xdist_auto_num_workers(config) -> Optional[int]:
    """Return the number of workers to use for pytest-xdist.

    This is used by pytest-xdist to automatically determine the number of
    workers to use. We want to use all available CPUs, but leave one CPU for
    the main process.
    """

    if config.option.numprocesses == "auto":
        return max(1, (available_cpus() - 1) // num_workers())
    return None


################################################################################
# CloudPickleCache Plugins


def pytest_configure(config: "Config"):
    config.cloud_pickle_cache = CloudPickleCache.for_config(config, _ispytest=True)

    config.addinivalue_line(
        "markers",
        "slow: mark a test as slow and only run if explicitly request with the --slow-tests flag",
    )

    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is not None:
        logging.basicConfig(
            format="%(asctime)s %(levelname)s %(message)s",
            filename=f"tests_{worker_id}.log",
            level=logging.DEBUG,
        )


def pytest_runtest_setup(item: pytest.Item):
    marker = item.get_closest_marker("slow")
    if marker:
        if not item.config.getoption("--slow-tests"):
            pytest.skip("slow test")


def is_osx_arm64():
    return platform.system() == "Darwin" and platform.machine() == "arm64"

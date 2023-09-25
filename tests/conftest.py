import os
from dataclasses import asdict
from typing import TYPE_CHECKING, Tuple

import numpy as np
import pytest
from pymemcache.client import Client
from sklearn import datasets
from sklearn.utils import Bunch

from pydvl.parallel.backend import available_cpus
from pydvl.utils import Dataset, MemcachedClientConfig
from tests.cache import CloudPickleCache
from tests.tolerate import (
    TolerateErrorFixture,
    TolerateErrorsSession,
    wrap_pytest_function,
)

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.fixtures import FixtureRequest
    from _pytest.terminal import TerminalReporter


def pytest_addoption(parser):
    parser.addoption(
        "--do-not-start-memcache",
        action="store_true",
        help="When this flag is used, memcache won't be started by a fixture"
        " and is instead expected to be already running",
    )
    group = parser.getgroup("tolerate")
    group.addoption(
        "--tolerate-verbose",
        action="store_true",
        default=False,
        help="Dump diagnostic and progress information.",
    )
    group.addoption(
        "--tolerate-quiet",
        action="store_true",
        default=False,
        help="Disable reporting. Verbose mode takes precedence.",
    )


@pytest.fixture
def cache(request: "FixtureRequest") -> CloudPickleCache:
    """Return a cache object that can persist state between testing sessions.

    cache.get(key, default)
    cache.set(key, value)

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
    try:
        client = Client(server=(hostname, port))
        client.flush_all()
        return True
    except ConnectionRefusedError:
        return False


@pytest.fixture(scope="session")
def do_not_start_memcache(request):
    return request.config.getoption("--do-not-start-memcache")


@pytest.fixture(scope="session")
def docker_services(
    docker_compose_command,
    docker_compose_file,
    docker_compose_project_name,
    docker_setup,
    docker_cleanup,
    do_not_start_memcache,
):
    """Start all services from a docker compose file (`docker-compose up`).
    After test are finished, shutdown all services (`docker-compose down`)."""
    from pytest_docker.plugin import get_docker_services

    if do_not_start_memcache:
        yield
    else:
        with get_docker_services(
            docker_compose_command,
            docker_compose_file,
            docker_compose_project_name,
            docker_setup,
            docker_cleanup,
        ) as docker_service:
            yield docker_service


@pytest.fixture(scope="session")
def memcached_service(docker_ip, docker_services, do_not_start_memcache):
    """Ensure that memcached service is up and responsive.

    If `do_not_start_memcache` is True then we just return the default values
    'localhost', 11211
    """
    if do_not_start_memcache:
        return "localhost", 11211
    else:
        # `port_for` takes a container port and returns the corresponding host port
        port = docker_services.port_for("memcached", 11211)
        hostname, port = docker_ip, port
        docker_services.wait_until_responsive(
            timeout=30.0,
            pause=0.5,
            check=lambda: is_memcache_responsive(hostname, port),
        )
        return hostname, port


@pytest.fixture(scope="function")
def memcache_client_config(memcached_service) -> MemcachedClientConfig:
    client_config = MemcachedClientConfig(
        server=memcached_service, connect_timeout=1.0, timeout=1, no_delay=True
    )
    Client(**asdict(client_config)).flush_all()
    return client_config


@pytest.fixture(scope="function")
def memcached_client(memcache_client_config) -> Tuple[Client, MemcachedClientConfig]:
    from pymemcache.client import Client

    try:
        c = Client(**asdict(memcache_client_config))
        c.flush_all()
        return c, memcache_client_config
    except Exception as e:
        print(
            f"Could not connect to memcached server "
            f'{memcache_client_config["server"]}: {e}'
        )
        raise e


@pytest.fixture(scope="function")
def housing_dataset(num_points, num_features) -> Dataset:
    dataset = datasets.fetch_california_housing()
    dataset.data = dataset.data[:num_points, :num_features]
    dataset.feature_names = dataset.feature_names[:num_features]
    dataset.target = dataset.target[:num_points]
    return Dataset.from_sklearn(dataset, train_size=0.5)


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


def num_workers():
    # Run with 2 CPUs inside GitHub actions
    if os.getenv("CI"):
        return 2
    # And a maximum of 4 CPUs locally (most tests don't really benefit from more)
    return max(1, min(available_cpus() - 1, 4))


@pytest.fixture(scope="session")
def n_jobs():
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
# Tolerate Errors and CloudPickleCache Plugins


def pytest_configure(config: "Config"):
    config.addinivalue_line(
        "markers",
        "tolerate: mark a test to swallow errors up to a certain threshold. "
        "Use to test (ε,δ)-approximations.",
    )
    config._tolerate_session = TolerateErrorsSession(config)
    config.cloud_pickle_cache = CloudPickleCache.for_config(config, _ispytest=True)


@pytest.fixture(scope="function")
def tolerate(request: pytest.FixtureRequest):
    fixture = TolerateErrorFixture(request.node)
    return fixture


def pytest_runtest_setup(item: pytest.Item):
    marker = item.get_closest_marker("tolerate")
    if marker:
        if not marker.kwargs:
            raise ValueError("tolerate marker requires keywords arguments")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Function):
    marker = item.get_closest_marker("tolerate")
    has_fixture = hasattr(item, "funcargs") and isinstance(
        item.funcargs.get("tolerate"), TolerateErrorFixture
    )
    if marker:
        if not has_fixture:
            wrap_pytest_function(item)
    yield


def pytest_terminal_summary(
    terminalreporter: "TerminalReporter", exitstatus: int, config: "Config"
):
    tolerate_session = terminalreporter.config._tolerate_session
    tolerate_session.display(terminalreporter)

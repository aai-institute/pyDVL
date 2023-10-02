import functools
import os
from collections import defaultdict
from dataclasses import asdict
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Type

import numpy as np
import pytest
from pymemcache.client import Client
from sklearn import datasets
from sklearn.utils import Bunch

from pydvl.parallel.backend import available_cpus
from pydvl.utils import Dataset, MemcachedClientConfig

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.terminal import TerminalReporter

EXCEPTIONS_TYPE = Optional[Sequence[Type[BaseException]]]


def is_memcache_responsive(hostname, port):
    try:
        client = Client(server=(hostname, port))
        client.flush_all()
        return True
    except ConnectionRefusedError:
        return False


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


@pytest.fixture(scope="session")
def do_not_start_memcache(request, worker_id):
    if worker_id != "master":
        return True
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

    :param a: Slope
    :param b: intercept
    :param num_points: number of (x,y) samples to construct
    :param train_size: fraction of points to use for training (between 0 and 1)

    :return: Dataset with train/test split. call str() on it to see the parameters
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
# Tolerate Errors Plugin


class TolerateErrorsSession:
    def __init__(self, config: "Config") -> None:
        self.verbose = config.getoption("tolerate_verbose")
        self.quiet = False if self.verbose else config.getoption("tolerate_quiet")
        self.columns = ["passed", "failed", "skipped", "max_failures"]
        self.labels = {
            "name": "Name",
            "passed": "Passed",
            "failed": "Failed",
            "skipped": "Skipped",
            "max_failures": "Maximum Allowed # Failures",
        }
        self._tests = defaultdict(TolerateErrorsTestItem)

    def get_max_failures(self, key: str) -> int:
        return self._tests[key].max_failures

    def set_max_failures(self, key: str, value: int) -> None:
        self._tests[key].max_failures = value

    def get_num_passed(self, key: str) -> int:
        return self._tests[key].passed

    def increment_num_passed(self, key: str) -> None:
        self._tests[key].passed += 1

    def get_num_failures(self, key: str) -> int:
        return self._tests[key].failed

    def increment_num_failures(self, key: str) -> None:
        self._tests[key].failed += 1

    def get_num_skipped(self, key: str) -> int:
        return self._tests[key].skipped

    def increment_num_skipped(self, key: str) -> None:
        self._tests[key].skipped += 1

    def set_exceptions_to_ignore(self, key: str, value: EXCEPTIONS_TYPE) -> None:
        if value is None:
            self._tests[key].exceptions_to_ignore = tuple()
        elif isinstance(value, Sequence):
            self._tests[key].exceptions_to_ignore = value
        else:
            self._tests[key].exceptions_to_ignore = (value,)

    def get_exceptions_to_ignore(self, key: str) -> EXCEPTIONS_TYPE:
        return self._tests[key].exceptions_to_ignore

    def has_exceeded_max_failures(self, key: str) -> bool:
        return self._tests[key].failed > self._tests[key].max_failures

    def display(self, terminalreporter: "TerminalReporter"):
        if self.quiet:
            return
        if len(self._tests) == 0:
            return
        terminalreporter.ensure_newline()
        terminalreporter.write_line("")
        widths = {
            "name": 3
            + max(len(self.labels["name"]), max(len(name) for name in self._tests))
        }
        for key in self.columns:
            widths[key] = 5 + len(self.labels[key])

        labels_line = self.labels["name"].ljust(widths["name"]) + "".join(
            self.labels[prop].rjust(widths[prop]) for prop in self.columns
        )
        terminalreporter.write_line(
            " tolerate: {count} tests ".format(count=len(self._tests)).center(
                len(labels_line), "-"
            ),
            yellow=True,
        )
        terminalreporter.write_line(labels_line)
        terminalreporter.write_line("-" * len(labels_line), yellow=True)
        for name in self._tests:
            has_error = self.has_exceeded_max_failures(name)
            terminalreporter.write(
                name.ljust(widths["name"]),
                red=has_error,
                green=not has_error,
                bold=True,
            )
            for prop in self.columns:
                terminalreporter.write(
                    "{0:>{1}}".format(self._tests[name][prop], widths[prop])
                )
            terminalreporter.write("\n")
        terminalreporter.write_line("-" * len(labels_line), yellow=True)
        terminalreporter.write_line("")


class TolerateErrorsTestItem:
    def __init__(self):
        self.max_failures = 0
        self.failed = 0
        self.passed = 0
        self.skipped = 0
        self.exceptions_to_ignore = tuple()

    def __getitem__(self, item: str):
        return getattr(self, item)


class TolerateErrorFixture:
    def __init__(self, node: pytest.Item):
        if hasattr(node, "originalname"):
            self.name = node.originalname
        else:
            self.name = node.name
        self.session: TolerateErrorsSession = node.config._tolerate_session
        marker = node.get_closest_marker("tolerate")
        if marker:
            max_failures = marker.kwargs.get("max_failures")
            exceptions_to_ignore = marker.kwargs.get("exceptions_to_ignore")
            self.session.set_max_failures(self.name, max_failures)
            self.session.set_exceptions_to_ignore(self.name, exceptions_to_ignore)

    def __call__(
        self, max_failures: int, *, exceptions_to_ignore: EXCEPTIONS_TYPE = None
    ):
        self.session.set_max_failures(self.name, max_failures)
        self.session.set_exceptions_to_ignore(self.name, exceptions_to_ignore)
        return self

    def __enter__(self):
        if self.session.has_exceeded_max_failures(self.name):
            self.session.increment_num_skipped(self.name)
            pytest.skip(
                f"Maximum number of allowed failures, {self.session.get_max_failures(self.name)}, was already exceeded"
            )

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.session.increment_num_passed(self.name)
        else:
            exceptions_to_ignore = self.session.get_exceptions_to_ignore(self.name)
            if not any(exc_type is x for x in exceptions_to_ignore):
                self.session.increment_num_failures(self.name)
        if self.session.has_exceeded_max_failures(self.name):
            pytest.fail(
                f"Maximum number of allowed failures, {self.session.get_max_failures(self.name)}, was exceeded"
            )
        return True


def wrap_pytest_function(pyfuncitem: pytest.Function):
    testfunction = pyfuncitem.obj
    tolerate_obj = TolerateErrorFixture(pyfuncitem)

    @functools.wraps(testfunction)
    def wrapper(*args, **kwargs):
        with tolerate_obj:
            testfunction(*args, **kwargs)

    pyfuncitem.obj = wrapper


@pytest.fixture(scope="function")
def tolerate(request: pytest.FixtureRequest):
    fixture = TolerateErrorFixture(request.node)
    return fixture


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "tolerate: mark a test to swallow errors up to a certain threshold. "
        "Use to test (ε,δ)-approximations.",
    )
    config._tolerate_session = TolerateErrorsSession(config)


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

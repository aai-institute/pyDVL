import functools
from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING, Optional, Sequence, Type

if TYPE_CHECKING:
    from _pytest.terminal import TerminalReporter
    from _pytest.config import Config

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from valuation.utils import Dataset, Utility
from valuation.utils.logging import start_logging_server
from valuation.utils.numeric import spearman

EXCEPTIONS_TYPE = Optional[Sequence[Type[BaseException]]]


def pytest_sessionstart():
    start_logging_server()


def is_memcache_responsive(hostname, port):
    from pymemcache.client import Client

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


@pytest.fixture(scope="session")
def do_not_start_memcache(request):
    return request.config.getoption("--do-not-start-memcache")


@pytest.fixture(scope="session")
def docker_services(
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
            docker_compose_file,
            docker_compose_project_name,
            docker_setup,
            docker_cleanup,
        ) as docker_service:
            yield docker_service


@pytest.fixture(scope="session")
def memcached_service(docker_ip, docker_services, do_not_start_memcache):
    """Ensure that memcached service is up and responsive.
    If do_not_start_memcache is True then we just return the default values: 'localhost', 11211
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


@pytest.fixture(scope="session")
def memcache_client_config(memcached_service):
    from valuation.utils import ClientConfig

    client_config = ClientConfig(
        server=memcached_service, connect_timeout=1.0, timeout=1, no_delay=True
    )
    return client_config


@pytest.fixture(scope="function")
def memcached_client(memcache_client_config):
    from pymemcache.client import Client

    try:
        c = Client(**memcache_client_config)
        c.flush_all()
        return c, memcache_client_config
    except Exception as e:
        print(
            f"Could not connect to memcached server "
            f'{memcache_client_config["server"]}: {e}'
        )
        raise e


@pytest.fixture(scope="function")
def boston_dataset(n_points, n_features):
    from sklearn import datasets

    dataset = datasets.load_boston()
    dataset.data = dataset.data[:n_points, :n_features]
    dataset.feature_names = dataset.feature_names[:n_features]
    dataset.target = dataset.target[:n_points]
    return Dataset.from_sklearn(dataset, train_size=0.5)


def polynomial(coefficients, x):
    powers = np.arange(len(coefficients))
    return np.power(x, np.tile(powers, (len(x), 1)).T).T @ coefficients


@pytest.fixture(scope="function")
def polynomial_dataset(coefficients: np.ndarray):
    """Coefficients must be for monomials of increasing degree"""
    from sklearn.utils import Bunch

    x = np.arange(-1, 1, 0.05)
    locs = polynomial(coefficients, x)
    y = np.random.normal(loc=locs, scale=0.3)
    db = Bunch()
    db.data, db.target = x.reshape(-1, 1), y
    poly = [f"{c} x^{i}" for i, c in enumerate(coefficients)]
    poly = " + ".join(poly)
    db.DESCR = f"$y \\sim N({poly}, 1)$"
    db.feature_names = ["x"]
    db.target_names = ["y"]
    return Dataset.from_sklearn(data=db, train_size=0.15), coefficients


@pytest.fixture(scope="function")
def polynomial_pipeline(coefficients):
    return make_pipeline(PolynomialFeatures(len(coefficients) - 1), LinearRegression())


@pytest.fixture(scope="function")
def linear_dataset(a, b, num_points):
    from sklearn.utils import Bunch

    step = 2 / num_points
    x = np.arange(-1, 1, step)
    y = np.random.normal(loc=a * x + b, scale=0.1)
    db = Bunch()
    db.data, db.target = x.reshape(-1, 1), y
    db.DESCR = f"y~N({a}*x + {b}, 1)"
    db.feature_names = ["x"]
    db.target_names = ["y"]
    return Dataset.from_sklearn(data=db, train_size=0.3)


def dummy_utility(num_samples: int = 10):
    from numpy import ndarray

    from valuation.utils import SupervisedModel

    # Indices match values
    x = np.arange(0, num_samples, 1).reshape(-1, 1)
    nil = np.zeros_like(x)
    data = Dataset(
        x, nil, nil, nil, feature_names=["x"], target_names=["y"], description=["dummy"]
    )

    class DummyModel(SupervisedModel):
        """Under this model each data point receives a score of index / max,
        assuming that the values of training samples match their indices."""

        def __init__(self, data: Dataset):
            self.m = max(data.x_train)
            self.utility = 0

        def fit(self, x: ndarray, y: ndarray):
            self.utility = np.sum(x) / self.m

        def predict(self, x: ndarray) -> ndarray:
            return x

        def score(self, x: ndarray, y: ndarray) -> float:
            return self.utility

    return Utility(DummyModel(data), data, scoring=None, enable_cache=False)


@pytest.fixture(scope="function")
def analytic_shapley(num_samples):
    """Scores are i/n, so v(i) = 1/n! Σ_π [U(S^π + {i}) - U(S^π)] = i/n"""
    u = dummy_utility(num_samples)
    exact_values = OrderedDict(
        {i: i / float(max(u.data.x_train)) for i in u.data.indices}
    )
    return u, exact_values


def check_total_value(u: Utility, values: OrderedDict, atol: float = 1e-6):
    """Checks absolute distance between total and added values.
    Shapley value is supposed to fulfill the total value axiom."""
    total_utility = u(u.data.indices)
    values = np.fromiter(values.values(), dtype=float, count=len(u.data))
    # We could want relative tolerances here if we didn't have the range of
    # the scorer.
    assert np.isclose(values.sum(), total_utility, atol=atol)


def check_exact(values: OrderedDict, exact_values: OrderedDict, atol: float = 1e-6):
    """Compares ranks and values."""

    k = list(values.keys())
    ek = list(exact_values.keys())

    assert np.all(k == ek), "Ranks do not match"

    v = np.array(list(values.values()))
    ev = np.array(list(exact_values.values()))

    assert np.allclose(v, ev, atol=atol)


def check_values(
    values: OrderedDict,
    exact_values: OrderedDict,
    rtol: float = 0.1,
    atol: float = 1e-5,
):
    """Compares value changes,
    without assuming keys in ordered dicts have the same order.

    Args:
        values:
        exact_values:
        rtol: relative tolerance of elements in values with respect to
            elements in exact values. E.g. if rtol = 0.1, we must have
            (values - exact_values)/exact_values < 0.1
    """
    for key in values:
        assert (
            abs(values[key] - exact_values[key]) < abs(exact_values[key]) * rtol + atol
        )


def check_rank_correlation(
    values: OrderedDict,
    exact_values: OrderedDict,
    k: int = None,
    threshold: float = 0.9,
):
    """Checks that the indices of `values` and `exact_values` follow the same
    order (by value), with some slack, using Spearman's correlation.

    Runs an assertion for testing.

    :param values: The values and indices to test
    :param exact_values: The ground truth
    :param k: Consider only these many, starting from the top.
    :param threshold: minimal value for spearman correlation for the test to
        succeed
    """
    # FIXME: estimate proper threshold for spearman
    if k is not None:
        raise NotImplementedError
    else:
        k = len(values)
    ranks = np.array(list(values.keys())[:k])
    ranks_exact = np.array(list(exact_values.keys())[:k])

    assert spearman(ranks, ranks_exact) >= threshold


# start_logging_server()


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
        if not self.quiet:
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
        self,
        max_failures: int,
        *,
        exceptions_to_ignore: EXCEPTIONS_TYPE = None,
    ):
        self.session.set_max_failures(self.name, max_failures)
        self.session.set_exceptions_to_ignore(self.name, exceptions_to_ignore)
        return self

    def __enter__(self):
        if self.session.has_exceeded_max_failures(self.name):
            self.session.increment_num_skipped(self.name)
            pytest.skip(
                f"Maximum number of allowed failures, {self.session.get_max_failures(self.name)}, was already reached"
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
                f"Maximum number of allowed failures, {self.session.get_max_failures(self.name)}, reached"
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
    fixture = TolerateErrorFixture(
        request.node,
    )
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


def pytest_terminal_summary(terminalreporter: "TerminalReporter"):
    tolerate_session = terminalreporter.config._tolerate_session
    tolerate_session.display(terminalreporter)

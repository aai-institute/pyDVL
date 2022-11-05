import functools
from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Type

import numpy as np
import pytest
import ray
from pymemcache.client import Client
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from pydvl.utils import Dataset, MemcachedClientConfig, Utility
from pydvl.utils.numeric import random_matrix_with_condition_number, spearman
from pydvl.utils.parallel import available_cpus

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.terminal import TerminalReporter

EXCEPTIONS_TYPE = Optional[Sequence[Type[BaseException]]]


@pytest.fixture(scope="session", autouse=True)
def ray_shutdown():
    yield
    ray.shutdown()


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


@pytest.fixture(autouse=True)
def pytorch_seed(seed):
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


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


@pytest.fixture(scope="function")
def memcache_client_config(memcached_service):

    client_config = MemcachedClientConfig(
        server=memcached_service, connect_timeout=1.0, timeout=1, no_delay=True
    )
    Client(**client_config).flush_all()
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


@pytest.fixture(scope="function")
def linear_dataset(a, b, num_points):
    from sklearn.utils import Bunch

    step = 2 / num_points
    stddev = 0.1
    x = np.arange(-1, 1, step)
    y = np.random.normal(loc=a * x + b, scale=stddev)
    db = Bunch()
    db.data, db.target = x.reshape(-1, 1), y
    db.DESCR = f"y~N({a}*x + {b}, {stddev:0.2f})"
    db.feature_names = ["x"]
    db.target_names = ["y"]
    return Dataset.from_sklearn(data=db, train_size=0.3)


def polynomial(coefficients, x):
    powers = np.arange(len(coefficients))
    return np.power(x, np.tile(powers, (len(x), 1)).T).T @ coefficients


@pytest.fixture
def input_dimension(request) -> int:
    return request.param


@pytest.fixture
def output_dimension(request) -> int:
    return request.param


@pytest.fixture
def problem_dimension(request) -> int:
    return request.param


@pytest.fixture
def batch_size(request) -> int:
    return request.param


@pytest.fixture
def condition_number(request) -> float:
    return request.param


@pytest.fixture(autouse=True)
def seed_numpy(seed=42):
    np.random.seed(seed)


@pytest.fixture
def num_workers():
    return max(1, available_cpus() - 1)


@pytest.fixture
def n_jobs(num_workers):
    return num_workers


@pytest.fixture(scope="function")
def quadratic_linear_equation_system(quadratic_matrix: np.ndarray, batch_size: int):
    A = quadratic_matrix
    problem_dimension = A.shape[0]
    b = np.random.random([batch_size, problem_dimension])
    return A, b


@pytest.fixture(scope="function")
def quadratic_matrix(problem_dimension: int, condition_number: float):
    return random_matrix_with_condition_number(problem_dimension, condition_number)


@pytest.fixture(scope="function")
def singular_quadratic_linear_equation_system(
    quadratic_matrix: np.ndarray, batch_size: int
):
    A = quadratic_matrix
    problem_dimension = A.shape[0]
    i, j = tuple(np.random.choice(problem_dimension, replace=False, size=2))
    if j < i:
        i, j = j, i

    v = (A[i] + A[j]) / 2
    A[i], A[j] = v, v
    b = np.random.random([batch_size, problem_dimension])
    return A, b


@pytest.fixture(scope="function")
def linear_model(problem_dimension: Tuple[int, int], condition_number: float):
    output_dimension, input_dimension = problem_dimension
    A = random_matrix_with_condition_number(
        max(input_dimension, output_dimension), condition_number
    )
    A = A[:output_dimension, :input_dimension]
    b = np.random.uniform(size=[output_dimension])
    return A, b


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


def dummy_utility(num_samples: int = 10):
    from numpy import ndarray

    from pydvl.utils import SupervisedModel

    # Indices match values
    x = np.arange(0, num_samples, 1).reshape(-1, 1)
    nil = np.zeros_like(x)
    data = Dataset(
        x, nil, nil, nil, feature_names=["x"], target_names=["y"], description="dummy"
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

    return Utility(DummyModel(data), data, enable_cache=False)


@pytest.fixture(scope="function")
def analytic_shapley(num_samples):
    """Scores are i/n, so v(i) = 1/n! Σ_π [U(S^π + {i}) - U(S^π)] = i/n"""
    u = dummy_utility(num_samples)
    exact_values = OrderedDict(
        {i: i / float(max(u.data.x_train)) for i in u.data.indices}
    )
    return u, exact_values


class TolerateErrors:
    """A context manager to swallow errors up to a certain threshold.
    Use to test (ε,δ)-approximations.
    """

    def __init__(
        self, max_errors: int, exception_cls: Type[BaseException] = AssertionError
    ):
        self.max_errors = max_errors
        self.Exception = exception_cls
        self.error_count = 0

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_count += 1
        if self.error_count > self.max_errors:
            raise self.Exception(
                f"Maximum number of {self.max_errors} error(s) reached"
            )
        return True


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

    assert np.allclose(v, ev, atol=atol), f"{v} != {ev}"


def check_values(
    values: Dict,
    exact_values: Dict,
    rtol: float = 0.1,
    atol: float = 1e-5,
):
    """Compares values in dictionaries.

    Asserts that `|value - exact_value| < |exact_value| * rtol + atol` for
    all pairs of `value`, `exact_value` with equal keys.

    Note that this does not assume any ordering (despite values typically being
    stored in an OrderedDict elsewhere.

    :param values:
    :param exact_values:
    :param rtol: relative tolerance of elements in `values` with respect to
        elements in `exact_values`. E.g. if rtol = 0.1, and atol = 0 we must
        have |value - exact_value|/|exact_value| < 0.1 for every value
    :param atol: absolute tolerance of elements in `values` with respect to
        elements in `exact_values`. E.g. if atol = 0.1, and rtol = 0 we must
        have |value - exact_value| < 0.1 for every value.
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

    correlation = spearman(ranks, ranks_exact)
    assert correlation >= threshold, f"{correlation} < {threshold}"


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


def pytest_terminal_summary(
    terminalreporter: "TerminalReporter", exitstatus: int, config: "Config"
):
    tolerate_session = terminalreporter.config._tolerate_session
    tolerate_session.display(terminalreporter)


def create_mock_dataset(
    linear_model: Tuple[np.ndarray, np.ndarray],
    train_set_size: int,
    test_set_size: int,
    noise: float = 0.01,
) -> Dataset:
    A, b = linear_model
    o_d, i_d = tuple(A.shape)
    data_model = lambda x: np.random.normal(x @ A.T + b, noise)

    x_train = np.random.uniform(size=[train_set_size, i_d])
    y_train = data_model(x_train)
    x_test = np.random.uniform(size=[test_set_size, i_d])
    y_test = data_model(x_test)
    dataset = Dataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        is_multi_output=True,
    )

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    dataset.x_train = scaler_x.fit_transform(dataset.x_train)
    dataset.y_train = scaler_y.fit_transform(dataset.y_train)
    dataset.x_test = scaler_x.transform(dataset.x_test)
    dataset.y_test = scaler_y.transform(dataset.y_test)
    return (x_train, y_train), (x_test, y_test)

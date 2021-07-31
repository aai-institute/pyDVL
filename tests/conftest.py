import numpy as np
import pytest

from sklearn.linear_model import LinearRegression
from typing import OrderedDict, Type
from valuation.shapley import combinatorial_exact_shapley
from valuation.utils import Dataset, Utility
from valuation.utils.numeric import spearman


@pytest.fixture(scope="function")
def memcached_client():
    from pymemcache.client import Client
    # TODO: read config from some place, start new server for tests
    memcache_config = dict(server='localhost:11211',
                           connect_timeout=1.0, timeout=0.1, no_delay=True)
    try:
        c = Client(**memcache_config)
        # FIXME: Careful! this will invalidate the cache. I should start a
        #  dedicated server
        c.flush_all()
        return c, memcache_config
    except Exception as e:
        print(f'Could not connect to memcached server '
              f'{memcache_config["server"]}: {e}')
        raise e


@pytest.fixture(scope="module")
def boston_dataset():
    from sklearn import datasets
    return Dataset.from_sklearn(datasets.load_boston())


@pytest.fixture(scope="module")
def linear_dataset():
    from sklearn.utils import Bunch
    a = 2
    b = 0
    x = np.arange(-1, 1, .15)
    y = np.random.normal(loc=a * x + b, scale=0.1)
    db = Bunch()
    db.data, db.target = x.reshape(-1, 1), y
    db.DESCR = f"y~N({a}*x + {b}, 1)"
    db.feature_names = ["x"]
    db.target_names = ["y"]
    return Dataset.from_sklearn(data=db, train_size=0.66)


def polynomial(coefficients, x):
    powers = np.arange(len(coefficients))
    return np.power(x, np.tile(powers, (len(x), 1)).T).T @ coefficients


@pytest.fixture(scope="module")
def polynomial_dataset(coefficients: np.ndarray):
    """ Coefficients must be for monomials of increasing degree """
    from sklearn.utils import Bunch

    x = np.arange(-1, 1, .1)
    locs = polynomial(coefficients, x)
    y = np.random.normal(loc=locs, scale=0.1)
    db = Bunch()
    db.data, db.target = x.reshape(-1, 1), y
    poly = [f"{c} x^{i}" for i, c in enumerate(coefficients)]
    poly = " + ".join(poly)
    db.DESCR = f"$y \sim N({poly}, 1)$"
    db.feature_names = ["x"]
    db.target_names = ["y"]
    return Dataset.from_sklearn(data=db, train_size=0.5)


@pytest.fixture()
def scoring():
    return 'r2'


@pytest.fixture()
def exact_shapley(linear_dataset, scoring):
    model = LinearRegression()
    u = Utility(model, linear_dataset, scoring)
    values_c = combinatorial_exact_shapley(u, progress=False)
    return u, values_c


class TolerateErrors:
    def __init__(self, max_errors: int, exception_cls: Type[BaseException]):
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
                f"Maximum number of {self.max_errors} error(s) reached")
        return True


def check_total_value(u: Utility,
                      values: OrderedDict,
                      rtol: float = 0.01):
    """ Checks absolute distance between total and added values.
     Shapley value is supposed to fulfill the total value axiom."""
    total_utility = u(u.data.indices)
    values = np.array(list(values.values()))
    # We use relative tolerances here because we don't have the range of the
    # scorer.
    assert np.isclose(values.sum(), total_utility, rtol=rtol)


def check_exact(values: OrderedDict, exact_values: OrderedDict, eps: float):
    """ Compares ranks and values. """

    k = list(values.keys())
    ek = list(exact_values.keys())

    assert np.all(k == ek)

    v = np.array(list(values.values()))
    ev = np.array(list(exact_values.values()))

    assert np.allclose(v, ev, atol=eps)


def check_rank_correlation(values: OrderedDict, exact_values: OrderedDict,
                           n: int = None, threshold: float = 0.9):
    # FIXME: estimate proper threshold for spearman
    if n is not None:
        raise NotImplementedError
    else:
        n = len(values)
    ranks = np.array(list(values.keys())[:n])
    ranks_exact = np.array(list(exact_values.keys())[:n])

    assert spearman(ranks, ranks_exact) >= threshold

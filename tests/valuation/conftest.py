import joblib
import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import Bunch

from pydvl.parallel import JoblibParallelBackend
from pydvl.utils import SupervisedModel
from pydvl.utils.caching import InMemoryCacheBackend
from pydvl.utils.status import Status
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.games import (
    AsymmetricVotingGame,
    Game,
    MinerGame,
    ShoesGame,
    SymmetricVotingGame,
)
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.scorers import SupervisedScorer
from pydvl.valuation.utility import ModelUtility
from pydvl.value.shapley.naive import combinatorial_exact_shapley

from ..conftest import num_workers
from . import polynomial


@pytest.fixture(scope="module")
def test_game(request) -> Game:
    name, kwargs = request.param
    if name == "miner":
        game = MinerGame(n_players=kwargs["n_players"])
    elif name == "shoes":
        game = ShoesGame(left=kwargs["left"], right=kwargs["right"])
    elif name == "symmetric-voting":
        game = SymmetricVotingGame(n_players=kwargs["n_players"])
    elif name == "asymmetric-voting":
        game = AsymmetricVotingGame()
    else:
        raise ValueError(f"Unknown game '{name}'")
    return game


@pytest.fixture(scope="function")
def polynomial_dataset(coefficients: np.ndarray):
    """Coefficients must be for monomials of increasing degree"""
    x = np.arange(-1, 1, 0.1)
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
def dummy_train_data(num_samples):
    """Training data used in everything that uses dummy_utility."""
    x = np.arange(0, num_samples, 1).reshape(-1, 1)
    nil = np.zeros_like(x)
    data = Dataset(x, nil, feature_names=["x"], target_names=["y"], description="dummy")
    return data


@pytest.fixture(scope="function")
def dummy_test_data(num_samples):
    """Test data used in everything that uses dummy_utility."""
    nil = np.zeros((num_samples, 1))
    data = Dataset(
        nil, nil, feature_names=["x"], target_names=["y"], description="dummy"
    )
    return data


@pytest.fixture(scope="function")
def dummy_utility(dummy_train_data, dummy_test_data):
    """Dummy utility for which we get analytical solutions for several methods."""
    # Indices match values
    data = dummy_train_data
    test_data = dummy_test_data

    class DummyModel(SupervisedModel):
        """Under this model each data point receives a score of index / max,
        assuming that the values of training samples match their indices.

        The utility of a set is the sum of the utilities.
        """

        def __init__(self, data: Dataset):
            self.m = max(data.x)
            self.utility = 0

        def fit(self, x: NDArray, y: NDArray):
            self.utility = np.sum(x) / self.m

        def predict(self, x: NDArray) -> NDArray:
            return x

        def score(self, x: NDArray, y: NDArray) -> float:
            return self.utility

    model = DummyModel(data)

    scorer = SupervisedScorer(
        model, test_data=test_data, default=0, range=(0, data.x.sum() / data.x.max())
    )

    return ModelUtility(
        DummyModel(data),
        scorer=scorer,
        catch_errors=False,
        show_warnings=True,
    )


@pytest.fixture(scope="function")
def analytic_shapley(dummy_utility, dummy_train_data):
    r"""Scores are i/n, so v(i) = 1/n! Σ_π [U(S^π + {i}) - U(S^π)] = i/n"""

    m = float(max(dummy_train_data.x))
    values = np.array([i / m for i in dummy_train_data.indices])
    result = ValuationResult(
        algorithm="exact",
        values=values,
        variances=np.zeros_like(values),
        data_names=dummy_train_data.indices,
        status=Status.Converged,
    )
    return dummy_utility, result


@pytest.fixture(scope="function")
def analytic_banzhaf(dummy_utility):
    r"""Scores are i/n, so
    v(i) = 1/2^{n-1} Σ_{S_{-i}} [U(S + {i}) - U(S)] = i/n
    """

    m = float(max(dummy_utility.data.x))
    values = np.array([i / m for i in dummy_utility.data.indices])
    result = ValuationResult(
        algorithm="exact",
        values=values,
        variances=np.zeros_like(values),
        data_names=dummy_utility.data.indices,
        status=Status.Converged,
    )
    return dummy_utility, result


@pytest.fixture(scope="function")
def linear_shapley(cache, linear_dataset, scorer, n_jobs):
    """This fixture makes use of the cache fixture to avoid recomputing
    exact shapley values for each test run."""
    args_hash = cache.hash_arguments(linear_dataset, scorer, n_jobs)
    u_cache_key = f"linear_shapley_u_{args_hash}"
    exact_values_cache_key = f"linear_shapley_exact_values_{args_hash}"
    try:
        u = cache.get(u_cache_key, None)
        exact_values = cache.get(exact_values_cache_key, None)
    except Exception:
        cache.clear_cache(cache._cachedir)
        raise

    if u is None:
        u = ModelUtility(
            LinearRegression(),
            data=linear_dataset,
            scorer=scorer,
        )
        exact_values = combinatorial_exact_shapley(u, progress=False, n_jobs=n_jobs)
        cache.set(u_cache_key, u)
        cache.set(exact_values_cache_key, exact_values)
    return u, exact_values


@pytest.fixture(scope="module")
def parallel_backend():
    with joblib.parallel_config(n_jobs=num_workers()):
        yield JoblibParallelBackend()


@pytest.fixture()
def cache_backend():
    cache = InMemoryCacheBackend()
    yield cache
    cache.clear()


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

    TODO: This is a duplicate of the fixture in the global conftest.py but using the
    Dataset object from pydvl.valuation
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

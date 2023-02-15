import numpy as np
import pytest
import ray
from numpy.typing import NDArray
from ray.cluster_utils import Cluster
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from pydvl.utils import Dataset, SupervisedModel, Utility
from pydvl.utils.config import ParallelConfig
from pydvl.utils.status import Status
from pydvl.value import ValuationResult

from . import polynomial


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
def dummy_utility(num_samples):
    # Indices match values
    x = np.arange(0, num_samples, 1).reshape(-1, 1)
    nil = np.zeros_like(x)
    data = Dataset(
        x, nil, nil, nil, feature_names=["x"], target_names=["y"], description="dummy"
    )

    class DummyModel(SupervisedModel):
        """Under this model each data point receives a score of index / max,
        assuming that the values of training samples match their indices.

        The utility of a set is the sum of the utilities.
        """

        def __init__(self, data: Dataset):
            self.m = max(data.x_train)
            self.utility = 0

        def fit(self, x: NDArray, y: NDArray):
            self.utility = np.sum(x) / self.m

        def predict(self, x: NDArray) -> NDArray:
            return x

        def score(self, x: NDArray, y: NDArray) -> float:
            return self.utility

    return Utility(
        DummyModel(data),
        data,
        score_range=(0, x.sum() / x.max()),
        catch_errors=False,
        show_warnings=True,
        enable_cache=False,
    )


@pytest.fixture(scope="function")
def analytic_shapley(dummy_utility):
    """Scores are i/n, so v(i) = 1/n! Σ_π [U(S^π + {i}) - U(S^π)] = i/n"""

    m = float(max(dummy_utility.data.x_train))
    values = np.array([i / m for i in dummy_utility.data.indices])
    result = ValuationResult(
        algorithm="exact",
        values=values,
        variances=np.zeros_like(values),
        data_names=dummy_utility.data.indices,
        status=Status.Converged,
    )
    return dummy_utility, result


@pytest.fixture(scope="module", params=["sequential", "ray-local", "ray-external"])
def parallel_config(request):
    if request.param == "sequential":
        pytest.skip("Skipping 'sequential' because it doesn't work with TMC")
        yield ParallelConfig(backend=request.param)
    elif request.param == "ray-local":
        yield ParallelConfig(backend="ray")
        ray.shutdown()
    elif request.param == "ray-external":
        # Starts a head-node for the cluster.
        cluster = Cluster(
            initialize_head=True,
            head_node_args={
                "num_cpus": 4,
            },
        )
        yield ParallelConfig(backend="ray", address=cluster.address)
        ray.shutdown()
        cluster.shutdown()

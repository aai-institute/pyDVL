import numpy as np
import pytest
from joblib import parallel_config
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

from pydvl.valuation.dataset import Dataset, GroupedDataset
from pydvl.valuation.methods import KNNShapleyValuation, ShapleyValuation
from pydvl.valuation.samplers import DeterministicUniformSampler
from pydvl.valuation.stopping import NoStopping
from pydvl.valuation.utility.knn import KNNClassifierUtility


@pytest.fixture(scope="module")
def data():
    return Dataset.from_sklearn(
        datasets.load_iris(),
        train_size=0.05,
        random_state=1234,
        stratify_by_target=True,
    )


def test_against_exact_shapley(data, n_jobs):
    model = KNeighborsClassifier(n_neighbors=5)
    train, test = data

    utility = KNNClassifierUtility(model=model, test_data=test, clone_before_fit=False)
    sampler = DeterministicUniformSampler()
    exact_valuation = ShapleyValuation(
        utility, sampler=sampler, is_done=NoStopping(), progress=False
    )
    with parallel_config(n_jobs=n_jobs):
        exact_result = exact_valuation.fit(train).result

    valuation = KNNShapleyValuation(model=model, test_data=test, progress=False)
    with parallel_config(n_jobs=n_jobs):
        results = valuation.fit(train).result

    eps = float(np.finfo(float).eps)
    np.testing.assert_allclose(  # type: ignore
        results.values, exact_result.values, atol=eps, rtol=eps
    )

    np.testing.assert_allclose(
        results.values, exact_result.values, atol=1e-10, rtol=1e-10
    )


def test_unsupported_grouped_dataset(data):
    train, test = data
    data_groups = np.zeros(len(train))
    grouped = GroupedDataset.from_dataset(train, data_groups)

    valuation = KNNShapleyValuation(
        model=KNeighborsClassifier(n_neighbors=1), test_data=test, progress=False
    )

    with pytest.raises(TypeError, match="GroupedDataset is not supported"):
        valuation.fit(grouped)

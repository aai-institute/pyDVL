import numpy as np
import pytest
from joblib import parallel_config
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

from pydvl.utils.dataset import Dataset as OldDataset
from pydvl.utils.utility import Utility as OldUtility
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods import DataShapleyValuation, KNNShapleyValuation
from pydvl.valuation.samplers import PermutationSampler
from pydvl.valuation.stopping import MinUpdates
from pydvl.valuation.utility import KNNClassifierUtility
from pydvl.value.shapley.knn import knn_shapley as old_knn_shapley


@pytest.fixture(scope="module")
def data():
    return Dataset.from_sklearn(
        datasets.load_iris(),
        train_size=0.05,
        random_state=1234,
        stratify_by_target=True,
    )


@pytest.fixture(scope="module")
def montecarlo_results(data):
    model = KNeighborsClassifier(n_neighbors=5)
    data_train, data_test = data
    utility = KNNClassifierUtility(model=model, test_data=data_test)
    sampler = PermutationSampler(seed=42)
    montecarlo_valuation = DataShapleyValuation(
        utility,
        sampler=sampler,
        is_done=MinUpdates(1000),
        progress=False,
    )
    return montecarlo_valuation.fit(data_train).values()


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_against_montecarlo(n_jobs, data, montecarlo_results):
    model = KNeighborsClassifier(n_neighbors=5)
    data_train, data_test = data
    utility = KNNClassifierUtility(model=model, test_data=data_test)
    valuation = KNNShapleyValuation(utility, progress=False)

    with parallel_config(n_jobs=n_jobs):
        results = valuation.fit(data_train).values()

    np.testing.assert_allclose(
        results.values, montecarlo_results.values, atol=1e-2, rtol=1e-2
    )


@pytest.mark.xfail(reason="Suspected bug in old implementation.")
def test_old_vs_new(seed, data):
    model = KNeighborsClassifier(n_neighbors=5)
    old_data = OldDataset.from_sklearn(
        datasets.load_iris(),
        train_size=0.05,
        random_state=seed,
        stratify_by_target=True,
    )
    old_u = OldUtility(model=model, data=old_data)
    old_values = old_knn_shapley(old_u, progress=False).values

    data_train, data_test = data
    utility = KNNClassifierUtility(model=model, test_data=data_test)
    new_valuation = KNNShapleyValuation(utility, progress=False)
    new_values = new_valuation.fit(data_train).values().values

    np.testing.assert_allclose(new_values, old_values, atol=1e-2, rtol=1e-2)

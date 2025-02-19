import numpy as np
import pytest
from joblib import parallel_config
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

from pydvl.valuation.dataset import Dataset, GroupedDataset
from pydvl.valuation.methods import KNNShapleyValuation, ShapleyValuation
from pydvl.valuation.samplers import PermutationSampler
from pydvl.valuation.stopping import MinUpdates
from pydvl.valuation.utility import KNNClassifierUtility


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
    montecarlo_valuation = ShapleyValuation(
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


def test_unsupported_grouped_dataset(data):
    train, test = data
    data_groups = np.zeros(len(train))
    grouped = GroupedDataset.from_dataset(train, data_groups)

    utility = KNNClassifierUtility(
        model=KNeighborsClassifier(n_neighbors=1), test_data=test
    )
    valuation = KNNShapleyValuation(utility, progress=False)

    with pytest.raises(TypeError, match="GroupedDataset is not supported"):
        valuation.fit(grouped)

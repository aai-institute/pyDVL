import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

from pydvl.utils.dataset import Dataset as OldDataset
from pydvl.utils.utility import Utility as OldUtility
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods import DataShapleyValuation, KNNShapleyValuation
from pydvl.valuation.samplers import PermutationSampler
from pydvl.valuation.stopping import MinUpdates
from pydvl.valuation.utility import KNNUtility
from pydvl.value.shapley.knn import knn_shapley as old_knn_shapley


def test_old_vs_new(seed):
    model = KNeighborsClassifier(n_neighbors=5)

    # calculate old values
    old_data = OldDataset.from_sklearn(
        datasets.load_iris(),
        train_size=0.05,
        random_state=seed,
        stratify_by_target=True,
    )
    old_u = OldUtility(model=model, data=old_data)
    old_values = old_knn_shapley(old_u, progress=False).values

    # calculate new values
    data_train, data_test = Dataset.from_sklearn(
        datasets.load_iris(),
        train_size=0.05,
        random_state=seed,
        stratify_by_target=True,
    )
    utility = KNNUtility(model=model, test_data=data_test)
    new_valuation = KNNShapleyValuation(utility, progress=False)
    new_values = new_valuation.fit(data_train).values().values

    # calculate shapley values without exploiting the knn structure
    sampler = PermutationSampler(seed=seed)
    montecarlo_valuation = DataShapleyValuation(
        utility,
        sampler=sampler,
        is_done=MinUpdates(1000),
        progress=False,
    )
    montecarlo_values = montecarlo_valuation.fit(data_train).values().values

    np.testing.assert_allclose(new_values, montecarlo_values, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(new_values, old_values, atol=1e-2, rtol=1e-2)

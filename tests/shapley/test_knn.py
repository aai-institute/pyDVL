import logging

import numpy as np
from sklearn import datasets
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier

from valuation.shapley import knn_shapley, permutation_montecarlo_shapley
from valuation.utils import Dataset, Utility

log = logging.getLogger(__name__)


def knn_loss_function(labels, predictions, n_classes=3):
    log.info(f"{predictions=}")
    if len(predictions[0]) < n_classes:
        raise RuntimeError("Found less classes than expected.")
    pred_proba = [predictions[i][label] for i, label in enumerate(labels)]
    return np.mean(pred_proba)


def test_knn_montecarlo_match(seed):

    data = Dataset.from_sklearn(datasets.load_iris(), random_state=seed)

    knn = KNeighborsClassifier(n_neighbors=5)

    knn_values = knn_shapley(data, knn, False)
    knn_keys = list(knn_values.keys())
    scorer = make_scorer(knn_loss_function, greater_is_better=True, needs_proba=True)

    utility = Utility(
        knn,
        data=data,
        scoring=scorer,
        show_warnings=False,
        enable_cache=False,
    )
    shapley_values, _ = permutation_montecarlo_shapley(
        utility,
        max_iterations=500,
        progress=False,
        num_workers=8,
    )
    shapley_keys = list(shapley_values.keys())
    log.info(f"{knn_keys=}")
    log.info(f"{shapley_keys=}")

    # will check only matching top elements since the scoring functions are not exactly the same
    top_knn = knn_keys[:3]
    top_montecarlo = shapley_keys[:10]
    assert np.all([k in top_montecarlo for k in top_knn])

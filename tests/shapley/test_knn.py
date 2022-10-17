import logging

import numpy as np
from sklearn import datasets
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier

from pydvl.shapley.knn import knn_shapley
from pydvl.shapley.naive import combinatorial_exact_shapley
from pydvl.utils import Dataset, Utility, available_cpus

log = logging.getLogger(__name__)


def test_knn_montecarlo_match(seed):
    data = Dataset.from_sklearn(
        datasets.load_iris(), train_size=0.05, random_state=seed, stratify=True
    )
    model = KNeighborsClassifier(n_neighbors=5)

    knn_values = knn_shapley(data, model, progress=False)
    knn_keys = list(knn_values.keys())

    def knn_loss_function(labels, predictions, n_classes=3):
        log.debug(f"{predictions=}")
        if len(predictions[0]) < n_classes:
            raise RuntimeError("Found less classes than expected.")
        pred_proba = [predictions[i][label] for i, label in enumerate(labels)]
        return np.mean(pred_proba)

    scorer = make_scorer(knn_loss_function, greater_is_better=True, needs_proba=True)

    utility = Utility(
        model,
        data=data,
        scoring=scorer,
        show_warnings=False,
        enable_cache=False,
    )
    exact_values = combinatorial_exact_shapley(
        utility, progress=False, n_jobs=min(len(data), available_cpus())
    )
    exact_keys = list(exact_values.keys())
    log.debug(f"{knn_keys=}")
    log.debug(f"{exact_keys=}")

    # will check only matching top elements since the scoring functions are not exactly the same
    top_knn = knn_keys[:2]
    top_montecarlo = exact_keys[:4]
    assert np.all([k in top_montecarlo for k in top_knn])

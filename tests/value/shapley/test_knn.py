import logging

import numpy as np
from sklearn import datasets
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsClassifier

from pydvl.parallel.backend import available_cpus
from pydvl.utils.dataset import Dataset
from pydvl.utils.score import Scorer
from pydvl.utils.utility import Utility
from pydvl.value.shapley.knn import knn_shapley
from pydvl.value.shapley.naive import combinatorial_exact_shapley

log = logging.getLogger(__name__)


def test_knn_montecarlo_match(seed):
    data = Dataset.from_sklearn(
        datasets.load_iris(),
        train_size=0.05,
        random_state=seed,
        stratify_by_target=True,
    )
    model = KNeighborsClassifier(n_neighbors=5)
    u = Utility(model=model, data=data)
    knn_values = knn_shapley(u, progress=False)

    def knn_loss_function(labels, predictions, n_classes=3):
        log.debug(f"{predictions=}")
        if len(predictions[0]) < n_classes:
            raise RuntimeError("Found less classes than expected.")
        pred_proba = [predictions[i][label] for i, label in enumerate(labels)]
        return np.mean(pred_proba)

    scorer = Scorer(
        make_scorer(knn_loss_function, greater_is_better=True, needs_proba=True),
        default=0.0,
        range=(0.0, 1.0),
    )

    utility = Utility(
        model, data=data, scorer=scorer, show_warnings=False, enable_cache=False
    )
    exact_values = combinatorial_exact_shapley(
        utility, progress=False, n_jobs=min(len(data), available_cpus())
    )

    # will check only matching top elements since the scoring functions are not
    # exactly the same
    knn_values.sort()
    exact_values.sort()
    top_knn = knn_values.indices[-2:]
    top_exact = exact_values.indices[-4:]
    assert np.all([k in top_exact for k in top_knn])

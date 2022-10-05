from collections import OrderedDict
from typing import Dict, Union

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from valuation.reporting.scores import sort_values
from valuation.utils import Dataset, maybe_progress

__all__ = ["compute_knn_shapley"]


def compute_knn_shapley(
    data: Dataset, model: KNeighborsClassifier, progress: bool = True
) -> "OrderedDict[int, float]":
    """Computes exact Shapley values for a KNN classifier

    :param data: split Dataset
    :param model: model to extract parameters from. The object will not be
        modified nor used other than to call get_params()
    :param progress: whether to display a progress bar
    """
    defaults: Dict[str, Union[int, str]] = {
        "algorithm": "ball_tree" if data.dim >= 20 else "kd_tree",
        "metric": "minkowski",
        "p": 2,
    }
    defaults.update(model.get_params())
    # HACK: NearestNeighbors doesn't support this. There will be more...
    del defaults["weights"]
    n_neighbors: int = int(defaults["n_neighbors"])
    defaults["n_neighbors"] = len(data)  # We want all training points sorted

    assert n_neighbors < len(data)
    # assert data.target_dim == 1

    nns = NearestNeighbors(**defaults).fit(data.x_train)
    # closest to farthest
    _, indices = nns.kneighbors(data.x_test)

    values = {i: 0.0 for i in data.indices}
    n = len(data)
    yt = data.y_train
    iterator = enumerate(zip(data.y_test, indices), start=1)
    for j, (y, ii) in maybe_progress(iterator, progress):
        value_at_x = int(yt[ii[-1]] == y) / n
        values[ii[-1]] += (value_at_x - values[ii[-1]]) / j
        for i in range(n - 2, n_neighbors, -1):  # farthest to closest
            value_at_x = (
                values[ii[i + 1]] + (int(yt[ii[i]] == y) - int(yt[ii[i + 1]] == y)) / i
            )
            values[ii[i]] += (value_at_x - values[ii[i]]) / j
        for i in range(n_neighbors, -1, -1):  # farthest to closest
            value_at_x = (
                values[ii[i + 1]]
                + (int(yt[ii[i]] == y) - int(yt[ii[i + 1]] == y)) / n_neighbors
            )
            values[ii[i]] += (value_at_x - values[ii[i]]) / j

    return sort_values(values)

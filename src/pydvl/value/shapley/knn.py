"""
This module contains Shapley computations for K-Nearest Neighbours.

(to do: implement approximate KNN computation for sublinear complexity)
"""

from typing import Dict, OrderedDict, Union

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from pydvl.reporting.scores import sort_values
from pydvl.utils import Dataset, maybe_progress

__all__ = ["knn_shapley"]


def knn_shapley(
    data: Dataset, model: KNeighborsClassifier, *, progress: bool = True
) -> OrderedDict[str, float]:
    """Computes exact Shapley values for a KNN classifier.

    This implements the method described in :footcite:t:`jia_efficient_2019a`.
    It exploits the local structure of K-Nearest Neighbours to reduce the number
    of calls to the utility function to a constant number per index, thus
    reducing computation time to $O(n)$.

    :param data: A :class:`pydvl.utils.dataset.Dataset` object with a training /
        test split.
    :param model: A KNN model to extract parameters from. The object will not be
        modified nor used other than to call
        `get_params() <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.get_params>`_
    :param progress: Whether to display a progress bar.

    :return: An
        `OrderedDict <https://docs.python.org/3/library/collections.html#collections.OrderedDict>`_
        of data indices and their values.

    .. rubric:: References

    .. footbibliography::

    .. versionadded:: 0.1.0

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

    return sort_values({data.data_names[i]: v for i, v in values.items()})

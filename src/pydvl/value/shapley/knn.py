"""
This module contains Shapley computations for K-Nearest Neighbours.

.. todo::
   Implement approximate KNN computation for sublinear complexity)
"""

from typing import Dict, Union

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from pydvl.utils import Utility, maybe_progress
from pydvl.value.results import ValuationResult, ValuationStatus

__all__ = ["knn_shapley"]


def knn_shapley(u: Utility, *, progress: bool = True) -> ValuationResult:
    """Computes exact Shapley values for a KNN classifier.

    This implements the method described in :footcite:t:`jia_efficient_2019a`.
    It exploits the local structure of K-Nearest Neighbours to reduce the number
    of calls to the utility function to a constant number per index, thus
    reducing computation time to $O(n)$.

    :param u: Utility with a KNN model to extract parameters from. The object
        will not be modified nor used other than to call `get_params()
        <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.get_params>`_
    :param progress: Whether to display a progress bar.
    :return: Object with the data values.
    :raises TypeError: If the model in the utility is not a `KNeighborsClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_

    .. versionadded:: 0.1.0

    """
    if not isinstance(u.model, KNeighborsClassifier):
        raise TypeError("KNN Shapley requires a K-Nearest Neighbours model")

    defaults: Dict[str, Union[int, str]] = {
        "algorithm": "ball_tree" if u.data.dim >= 20 else "kd_tree",
        "metric": "minkowski",
        "p": 2,
    }
    defaults.update(u.model.get_params())
    # HACK: NearestNeighbors doesn't support this. There will be more...
    del defaults["weights"]
    n_neighbors: int = int(defaults["n_neighbors"])
    defaults["n_neighbors"] = len(u.data)  # We want all training points sorted

    assert n_neighbors < len(u.data)
    # assert data.target_dim == 1

    nns = NearestNeighbors(**defaults).fit(u.data.x_train)
    # closest to farthest
    _, indices = nns.kneighbors(u.data.x_test)

    values = np.zeros_like(u.data.indices, dtype=np.float_)
    n = len(u.data)
    yt = u.data.y_train
    iterator = enumerate(zip(u.data.y_test, indices), start=1)
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

    return ValuationResult(
        algorithm="knn_shapley",
        status=ValuationStatus.Converged,
        values=values,
        stderr=None,
        data_names=u.data.data_names,
    )

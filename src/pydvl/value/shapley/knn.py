"""
This module contains Shapley computations for K-Nearest Neighbours.

!!! Todo
    Implement approximate KNN computation for sublinear complexity


## References

[^1]: <a name="jia_efficient_2019a"></a>Jia, R. et al., 2019. [Efficient
    Task-Specific Data Valuation for Nearest Neighbor
    Algorithms](https://doi.org/10.14778/3342263.3342637). In: Proceedings of
    the VLDB Endowment, Vol. 12, No. 11, pp. 1610–1623.

"""

from typing import Dict, Union

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from tqdm.auto import tqdm

from pydvl.utils import Utility
from pydvl.utils.status import Status
from pydvl.value.result import ValuationResult

__all__ = ["knn_shapley"]


def knn_shapley(u: Utility, *, progress: bool = True) -> ValuationResult:
    """Computes exact Shapley values for a KNN classifier.

    This implements the method described in (Jia, R. et al., 2019)<sup><a
    href="#jia_efficient_2019a">1</a></sup>. It exploits the local structure of
    K-Nearest Neighbours to reduce the value computation to sorting of the training
    points by distance to the test point and applying a recursive formula,
    thus reducing computation time to $O(n_test n_train log(n_train)$.

    Args:
        u: Utility with a KNN model to extract parameters from. The object
            will not be modified nor used other than to call [get_params()](
            <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.get_params>)
        progress: Whether to display a progress bar.

    Returns:
        Object with the data values.

    Raises:
        TypeError: If the model in the utility is not a
            [sklearn.neighbors.KNeighborsClassifier][].

    !!! tip "New in version 0.1.0"

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

    res = np.zeros_like(u.data.indices, dtype=np.float64)
    n = len(u.data)
    yt = u.data.y_train
    iterator = enumerate(zip(u.data.y_test, indices), start=1)
    for j, (y, ii) in tqdm(iterator, disable=not progress):
        values = np.zeros_like(u.data.indices, dtype=np.float64)
        idx = ii[-1]
        values[idx] = int(yt[idx] == y) / n

        for i in range(n - 1, 0, -1):
            prev_idx = idx
            idx = ii[i - 1]
            values[idx] = values[prev_idx] + (
                int(yt[idx] == y) - int(yt[prev_idx] == y)
            ) / max(n_neighbors, i)
        res += values

    return ValuationResult(
        algorithm="knn_shapley",
        status=Status.Converged,
        values=res,
        data_names=u.data.data_names,
    )

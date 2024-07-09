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
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from typing_extensions import Self

from pydvl.utils.status import Status
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.utility import KNNClassifierUtility


class KNNShapleyValuation(Valuation):
    """Computes exact Shapley values for a KNN classifier.

    This implements the method described in
    (Jia, R. et al., 2019)<sup><a href="#jia_efficient_2019a">1</a></sup>.
    It exploits the local structure of K-Nearest Neighbours to reduce the number
    of calls to the utility function to a constant number per index, thus
    reducing computation time to $O(n)$.

    Args:
        utility: KNNUtility with a KNN model to extract parameters from. The object
            will not be modified nor used other than to call [get_params()](
            <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator.get_params>)
        progress: Whether to display a progress bar.

    """

    def __init__(self, utility: KNNClassifierUtility, progress: bool = True):
        self.utility = utility
        self.progress = progress

        config = self.utility.model.get_params()
        self.n_neighbors = config["n_neighbors"]

        del config["weights"]
        self.helper_model = NearestNeighbors(**config)

    def fit(self, data: Dataset) -> Self:
        """Calculate exact shapley values for a KNN model on a dataset.

        This fit method bypasses direct evaluations of the utility function and
        calculates the Shapley values directly.

        In contrast to other data valuation models, the runtime increases linearly
        with the size of the test dataset.

        Calculating the least core valuation is a computationally expensive task that
        can be parallelized. To do so, call the `fit()` method inside a
        `joblib.parallel_config` context manager as follows:

        ```python
        from joblib import parallel_config

        with parallel_config(n_jobs=4):
            valuation.fit(data)
        ```

        """
        self.helper_model = self.helper_model.fit(data.x)
        n_obs = len(data.x)
        n_test = len(self.utility.test_data)

        generator = zip(self.utility.test_data.x, self.utility.test_data.y)

        generator_with_progress = tqdm(
            generator,
            total=n_test,
            disable=not self.progress,
            position=0,
        )

        with Parallel(return_as="generator") as parallel:
            results = parallel(
                delayed(_compute_values_for_one_test_point)(
                    self.helper_model, x, y, data.y
                )
                for x, y in generator_with_progress
            )
            values = np.zeros(n_obs)
            for res in results:
                values += res
            values /= n_test

        res = ValuationResult(
            algorithm="knn_shapley",
            status=Status.Converged,
            values=values,
            data_names=data.data_names,
        )

        self.result = res
        return self


def _compute_values_for_one_test_point(
    helper_model: NearestNeighbors, x: NDArray, y: int, y_train: NDArray
) -> np.ndarray:
    """Compute the Shapley value for a single test data point.

    The shapley values of the whole test set are the average of the shapley values
    of the single test data points.

    Args:
        helper_model: A fitted NearestNeighbors model.
        x: A single test data point.
        y: The correct label of the test data point.
        y_train: The training labels.

    Returns:
        The Shapley values for the test data point.

    """
    n_obs = len(y_train)
    k = helper_model.get_params()["n_neighbors"]

    # sorts data indices from close to far
    sorted_indices = helper_model.kneighbors(
        x.reshape(1, -1), n_neighbors=n_obs, return_distance=False
    )[0]

    values = np.zeros(n_obs)

    idx = sorted_indices[-1]
    values[idx] = float(y_train[idx] == y) / n_obs
    # reverse range because we want to go from far to close
    for i in range(n_obs - 1, 0, -1):
        prev_idx = sorted_indices[i]
        idx = sorted_indices[i - 1]
        values[idx] = values[prev_idx]
        values[idx] += (int(y_train[idx] == y) - int(y_train[prev_idx] == y)) / max(
            k, i
        )

    return values

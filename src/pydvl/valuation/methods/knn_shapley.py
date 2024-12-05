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
from joblib.parallel import Parallel, delayed, get_active_backend
from more_itertools import chunked
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from typing_extensions import Self

from pydvl.utils.status import Status
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset, GroupedDataset
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
        super().__init__()
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
        with the size of the dataset.

        Calculating the KNN valuation is a computationally expensive task that
        can be parallelized. To do so, call the `fit()` method inside a
        `joblib.parallel_config` context manager as follows:

        ```python
        from joblib import parallel_config

        with parallel_config(n_jobs=4):
            valuation.fit(data)
        ```

        """

        if isinstance(data, GroupedDataset):
            raise TypeError("GroupedDataset is not supported by KNNShapleyValuation")

        x_train, y_train = data.data()
        self.helper_model = self.helper_model.fit(x_train)
        n_test = len(self.utility.test_data)

        _, n_jobs = get_active_backend()
        batch_size = (n_test // n_jobs) + (1 if n_test % n_jobs else 0)
        x_test, y_test = self.utility.test_data.data()
        generator = zip(chunked(x_test, batch_size), chunked(y_test, batch_size))
        generator_with_progress = tqdm(
            generator,
            total=n_test,
            disable=not self.progress,
            position=0,
        )

        with Parallel(return_as="generator_unordered") as parallel:
            results = parallel(
                delayed(self._compute_values_for_test_points)(
                    self.helper_model, np.array(x_test), np.array(y_test), y_train
                )
                for x_test, y_test in generator_with_progress
            )
            values = np.zeros(len(data))
            for res in results:
                values += res
            values /= n_test

        res = ValuationResult(
            algorithm="knn_shapley",
            status=Status.Converged,
            values=values,
            data_names=data.names,
        )

        self.result = res
        return self

    @staticmethod
    def _compute_values_for_test_points(
        helper_model: NearestNeighbors,
        x_test: NDArray,
        y_test: NDArray,
        y_train: NDArray,
    ) -> np.ndarray:
        """Compute the Shapley value using a set of test points.

        The Shapley value for a training point is computed over the whole test set by
        averaging the Shapley values of the single test data points.

        Args:
            helper_model: A fitted NearestNeighbors model.
            x_test: The test data points.
            y_test: The test labels.
            y_train: The labels for the training points to be valued.

        Returns:
            The Shapley values for the training data points.

        """
        n_obs = len(y_train)
        n_neighbors = helper_model.get_params()["n_neighbors"]

        # sorts data indices from close to far
        if x_test.ndim == 1:
            x_test = x_test.reshape(1, -1)

        sorted_indices = helper_model.kneighbors(
            x_test, n_neighbors=n_obs, return_distance=False
        )

        values = np.zeros(shape=(len(x_test), n_obs))

        for query, neighbors in enumerate(sorted_indices):
            label = y_test[query]
            # Initialize the farthest neighbor's value
            idx = neighbors[-1]
            values[query][idx] = float(y_train[idx] == label) / n_obs
            # reverse range because we want to go from far to close
            for i in range(n_obs - 1, 0, -1):
                prev_idx = neighbors[i]
                idx = neighbors[i - 1]
                values[query][idx] = values[query][prev_idx]
                values[query][idx] += (
                    int(y_train[idx] == label) - int(y_train[prev_idx] == label)
                ) / max(n_neighbors, i)
                # 1/max(K, i) = 1/K * min{K, i}/i as in the paper

        return values.sum(axis=0)

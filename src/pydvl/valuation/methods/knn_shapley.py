r"""
This module contains Shapley computations for K-Nearest Neighbours classifier,
introduced by Jia et al. (2019).[^1]

In particular it provides
[KNNShapleyValuation][pydvl.valuation.methods.knn_shapley.KNNShapleyValuation] to
compute exact Shapley values for a KNN classifier in
$O(n \log n)$ time per test point, as opposed to $O(n^2 \log^2 n)$ if the model were
simply fed to a generic [ShapleyValuation][pydvl.valuation.methods.shapley.ShapleyValuation]
object.

See [the documentation][knn-shapley-intro] or the paper for details.

!!! Todo
    Implement approximate KNN computation for sublinear complexity


## References

[^1]: <a name="jia_efficient_2019a"></a>Jia, R. et al., 2019. [Efficient
    Task-Specific Data Valuation for Nearest Neighbor
    Algorithms](https://doi.org/10.14778/3342263.3342637). In: Proceedings of
    the VLDB Endowment, Vol. 12, No. 11, pp. 1610–1623.

"""

from __future__ import annotations

from typing import cast

import numpy as np
from joblib.parallel import Parallel, delayed, get_active_backend
from more_itertools import chunked
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from tqdm.auto import tqdm
from typing_extensions import Self

from pydvl.utils.status import Status
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset, GroupedDataset
from pydvl.valuation.result import ValuationResult


class KNNShapleyValuation(Valuation):
    """Computes exact Shapley values for a KNN classifier.

    This implements the method described in
    (Jia, R. et al., 2019)<sup><a href="#jia_efficient_2019a">1</a></sup>.

    Args:
        model: KNeighborsClassifier model to use for valuation
        test_data: Dataset containing test data to evaluate the model.
        progress: Whether to display a progress bar.
        clone_before_fit: Whether to clone the model before fitting.
    """

    algorithm_name: str = "knn_shapley"

    def __init__(
        self,
        model: KNeighborsClassifier,
        test_data: Dataset,
        progress: bool = True,
        clone_before_fit: bool = True,
    ):
        super().__init__()
        if not isinstance(model, KNeighborsClassifier):
            raise TypeError("KNN Shapley requires a K-Nearest Neighbours model")
        self.model = model
        self.test_data = test_data
        self.progress = progress
        self.clone_before_fit = clone_before_fit

    def fit(self, data: Dataset, continue_from: ValuationResult | None = None) -> Self:
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
        Args:
            data: The dataset to use for valuation.
            continue_from: A previously saved valuation result to continue from.

        """
        if isinstance(data, GroupedDataset):
            raise TypeError("GroupedDataset is not supported by KNNShapleyValuation")

        self._result = self._init_or_check_result(data, continue_from)

        x_train, y_train = data.data()
        if self.clone_before_fit:
            self.model = cast(KNeighborsClassifier, clone(self.model))
        self.model.fit(x_train, y_train)

        n_test = len(self.test_data)

        _, n_jobs = get_active_backend()
        n_jobs = n_jobs or 1  # Handle None if outside a joblib context
        batch_size = (n_test // n_jobs) + (1 if n_test % n_jobs else 0)
        x_test, y_test = self.test_data.data()
        batches = zip(chunked(x_test, batch_size), chunked(y_test, batch_size))

        process = delayed(self._compute_values_for_test_points)
        with Parallel(return_as="generator_unordered") as parallel:
            results = parallel(
                process(self.model, x_test, y_test, y_train)
                for x_test, y_test in batches
            )
            values = np.zeros(len(data))
            # FIXME: this progress bar won't add much since we have n_jobs batches and
            #  they will all take about the same time
            for res in tqdm(results, total=n_jobs, disable=not self.progress):
                values += res
            values /= n_test

        self._result += ValuationResult(
            algorithm=str(self),
            status=Status.Converged,
            values=values,
            data_names=data.names,
        )

        return self

    @staticmethod
    def _compute_values_for_test_points(
        model: NearestNeighbors,
        x_test: NDArray,
        y_test: NDArray,
        y_train: NDArray,
    ) -> NDArray[np.float64]:
        """Compute the Shapley value using a set of test points.

        The Shapley value for a training point is computed over the whole test set by
        averaging the Shapley values of the single test data points.

        Args:
            model: A fitted NearestNeighbors model.
            x_test: The test data points.
            y_test: The test labels.
            y_train: The labels for the training points to be valued.

        Returns:
            The Shapley values for the training data points.

        """
        n_obs = len(y_train)
        n_neighbors = model.get_params()["n_neighbors"]

        # sort data indices from close to far
        sorted_indices = model.kneighbors(
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

        return cast(NDArray[np.float64], values.sum(axis=0))

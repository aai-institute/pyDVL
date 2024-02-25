"""
!!! tip "New in version 0.9.0"

## References

[^1]: <a name="just_lava_2023"></a>Just et al.
[LAVA: Data Valuation without Pre-Specified Learning Algorithms](https://arxiv.org/abs/2305.00054).
In: Published at ICRL 2023
"""

import itertools
from typing import List, Tuple

import numpy as np
import ot
from numpy.typing import NDArray
from ot.gaussian import bures_wasserstein_distance

from pydvl.utils.dataset import Dataset

__all__ = ["LAVA"]


class LAVA:
    """Computes Data values using LAVA.

    This implements the method described in
    (Just et al., 2023)<sup><a href="just_lava_2023">1</a></sup>.

    Args:
        dataset: The dataset containing training and test samples.
        regularization: Regularization parameter for sinkhorn iterations.
        lambda_: Weight parameter for label distances.

    Examples:
        >>> from pydvl.lava import LAVA
        >>> from pydvl.utils.dataset import Dataset
        >>> from sklearn.datasets import load_iris
        >>> dataset = Dataset.from_sklearn(load_iris())
        >>> lava = LAVA(dataset)
        >>> values = lava.compute_values()
        >>> assert values.shape == (len(dataset.x_train),)
    """

    def __init__(
        self, dataset: Dataset, *, regularization: float = 0.001, lambda_: float = 1.0
    ) -> None:
        self.dataset = dataset
        self.regularization = regularization
        self.lambda_ = lambda_

    def compute_values(self) -> NDArray:
        """Compute calibrated gradients using Optimal Transport.

        Returns:
            Array of dimensions `(n_train)` that contains the calibrated gradients.
        """
        dual_solution = self._compute_ot_dual()
        calibrated_gradients = self._compute_calibrated_gradients(dual_solution)
        return calibrated_gradients

    def _compute_calibrated_gradients(
        self, dual_solution: Tuple[NDArray, NDArray]
    ) -> NDArray:
        """Compute calibrated gradients using the dual solution of Optimal Transport problem.

        $$
        \frac{\partial OT(\mu_t, \mu_v )}{\partial \mu_t(z_i)}
        = f_i^∗ - \sum\limits_{j\in\{1,\dots,N\} \setminus i} \frac{f_j^∗}{N-1}
        $$

        Args:
            dual_solution: Dual solution of Optimal Transport problem.

        Returns:
            Array of dimensions `(n_train)`
        """
        f1k = np.array(dual_solution[0])
        training_size = len(self.dataset.x_train)
        calibrated_gradients = (1 + 1 / (training_size - 1)) * f1k - sum(f1k) / (
            training_size - 1
        )
        return calibrated_gradients

    def _compute_ot_dual(self) -> Tuple[NDArray, NDArray]:
        """Compute the dual solution of the Optimal Transport problem."""
        label_distances = self._compute_label_distances()
        feature_distances = self._compute_feature_distances()
        M = (
            label_distances.shape[1] * self.dataset.y_train[..., np.newaxis]
            + self.dataset.y_test[np.newaxis, ...]
        )
        label_cost = label_distances.ravel()[M.ravel()].reshape(
            len(self.dataset.y_train), len(self.dataset.y_test)
        )
        ground_cost = feature_distances + self.lambda_ * label_cost
        a = ot.unif(len(self.dataset.x_train))
        b = ot.unif(len(self.dataset.x_test))
        _, log = ot.sinkhorn(a, b, ground_cost, self.regularization, log=True)
        u, v = log["u"], log["v"]
        return u, v

    def _compute_feature_distances(self, p: int = 2) -> NDArray:
        """Compute distance between the features of the training and test sets.

        The first has dimensions `(n1, d1)` and the second has dimensions `(n2, d2)`.

        Args:
            p: p-norm

        Returns:
             Array with dimensions `(n1, n2)`
        """
        if p == 1:
            distance = ot.dist(
                self.dataset.x_train, self.dataset.x_test, metric="euclidean"
            )
        elif p == 2:
            distance = ot.dist(
                self.dataset.x_train, self.dataset.x_test, metric="sqeuclidean"
            )
        else:
            raise ValueError(f"Unsupported p value {p}")
        return distance

    def _compute_label_distances(self) -> NDArray:
        """Compute distances between classes in the training and test sets.

        The number of classes in the first set is `n_classes1` and the second is `n_classes2`.

        Returns:
            An array with dimensions `(n_classes1, n_classes2)`
        """
        means_train, covariances_train = self._compute_label_stats(
            self.dataset.x_train, self.dataset.y_train
        )
        means_test, covariances_test = self._compute_label_stats(
            self.dataset.x_test, self.dataset.y_test
        )

        n_train, n_test = len(means_train), len(means_test)

        D_train_test = np.zeros((n_train, n_test))
        for i, j in itertools.product(range(n_train), range(n_test)):
            distance = bures_wasserstein_distance(
                means_train[i],
                means_test[j],
                covariances_train[i],
                covariances_test[j],
            )
            D_train_test[i, j] = distance

        D_train_train = np.zeros((n_train, n_train))
        for i, j in itertools.combinations(range(n_train), 2):
            distance = bures_wasserstein_distance(
                means_train[i],
                means_train[j],
                covariances_train[i],
                covariances_train[j],
            )
            D_train_train[i, j] = D_train_train[j, i] = distance

        D_test_test = np.zeros((n_test, n_test))
        for i, j in itertools.combinations(range(n_train), 2):
            distance = bures_wasserstein_distance(
                means_test[i],
                means_test[j],
                covariances_test[i],
                covariances_test[j],
            )
            D_test_test[i, j] = D_test_test[j, i] = distance

        label_distances = np.concatenate(
            [
                np.concatenate([D_train_train, D_train_test], axis=1),
                np.concatenate([D_train_test, D_test_test], axis=1),
            ],
            axis=0,
        )

        return label_distances

    def _compute_label_stats(
        self,
        x: NDArray,
        y: NDArray,
    ) -> Tuple[List[NDArray], List[NDArray]]:
        """Compute means and covariances for each class.

        Args:
            x: Input data.
            y: Target labels.

        Returns:
            Means and covariances.
        """
        classes = np.sort(np.unique(y))
        means = []
        covariances = []
        for i, c in enumerate(classes):
            class_indices = np.where(y == c)
            filtered_x = x[class_indices]
            mean: NDArray = np.mean(filtered_x, axis=0).ravel()
            cov = np.cov(filtered_x.T)
            means.append(mean)
            covariances.append(cov)
        return means, covariances

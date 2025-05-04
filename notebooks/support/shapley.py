from typing import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = ["decision_boundary_fixed_variance_2d"]


def decision_boundary_fixed_variance_2d(
    mu_1: NDArray, mu_2: NDArray
) -> Callable[[NDArray], NDArray]:
    """
    Closed-form solution for decision boundary dot(a, b) + b = 0 with fixed variance.

    Args:
        mu_1: First mean.
        mu_2: Second mean.

    Returns:
        A callable which converts a continuous line (-infty, infty) to the decision boundary in feature space.
    """
    a = np.asarray([[0, 1], [-1, 0]]) @ (mu_2 - mu_1)
    b = (mu_1 + mu_2) / 2
    a = a.reshape([1, -1])
    return lambda z: z.reshape([-1, 1]) * a + b  # type: ignore

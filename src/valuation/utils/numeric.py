import numpy as np


def vanishing_derivatives(values: np.ndarray,
                          min_values: int,
                          value_tolerance: float):
    """Checks empirical convergence of the derivatives of rows to zero
    and returns the number of rows that have converged. """
    last_values = values[:, -min_values - 1:]
    d = np.diff(last_values, axis=1)
    zeros = np.isclose(d, 0.0, atol=value_tolerance).sum(axis=1)
    return np.sum(zeros >= min_values / 2)

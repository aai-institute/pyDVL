import numpy as np

from pydvl.value import ValuationResult, ValuationStatus
from pydvl.value.shapley.semivalues import (
    beta_shapley,
    beta_shapley_paper,
    permutation_shapley,
    shapley,
    variance_criterion,
)
from value import check_values
import pytest


@pytest.mark.parametrize("fun", [beta_shapley, beta_shapley_paper])
@pytest.mark.parametrize("num_samples, max_subsets", [(10, 1000)])
def test_beta_shapley(analytic_shapley, fun, max_subsets):
    u, exact_values = analytic_shapley
    values = fun(u, alpha=1, beta=1, max_subsets=int(max_subsets))
    check_values(values, exact_values, rtol=0.1)


@pytest.mark.parametrize(
    "num_samples, fun, max_samples",
    [(8, shapley, 2**12), (10, permutation_shapley, 300)],
)
def test_shapley(analytic_shapley, fun, max_samples):
    u, exact_values = analytic_shapley

    values = fun(u, int(max_samples))
    check_values(values, exact_values, rtol=0.1)


@pytest.mark.parametrize(
    "values, variances, counts, eps, values_ratio, status",
    [
        ([1, 1, 1], [0, 0, 0], [1, 1, 1], 1e-4, 1.0, ValuationStatus.Converged),
        ([1, 2, 3], [0.01, 0.03, 0.08], [1, 1, 1], 0.1, 1.0, ValuationStatus.Converged),
        ([1, 2, 3], [0.01, 0.03, 0.8], [1, 1, 1], 0.1, 1.0, ValuationStatus.Pending),
        (
            [1, 2, 3],
            [0.01, 0.03, 0.8],
            [1, 1, 1],
            0.1,
            2 / 3,
            ValuationStatus.Converged,
        ),
    ],
)
def test_variance_criterion(values, variances, counts, eps, values_ratio, status):
    assert (
        variance_criterion(
            np.array(values, dtype=float),
            np.array(variances, dtype=float),
            np.array(counts, dtype=int),
            eps,
            values_ratio,
        )
        == status
    )


@pytest.mark.parametrize(
    "values, variances, counts, exception",
    [
        (np.array([]), np.array([]), np.array([]), ValueError),
        (np.array([1.0]), np.array([]), np.array([]), ValueError),
        (np.array([1.0]), np.array([1.0]), np.array([]), ValueError),
        (np.array([1.0]), np.array([]), np.array([1]), ValueError),
        ([1], [1], [1], TypeError),
    ],
)
def test_variance_criterion_exceptions(values, variances, counts, exception):
    with pytest.raises(exception):
        variance_criterion(values, variances, counts, 0.1, 1.0)

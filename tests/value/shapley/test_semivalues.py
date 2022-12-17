import math

import numpy as np
import pytest
from value import check_values

from pydvl.value import ValuationStatus
from pydvl.value.shapley.semivalues import (
    Coefficient,
    SemiValue,
    StoppingCriterion,
    banzhaf_coefficient,
    beta_coefficient,
    beta_shapley,
    beta_shapley_paper,
    combinatorial_coefficient,
    max_samples_criterion,
    permutation_shapley,
    shapley,
    stderr_criterion,
)


@pytest.mark.parametrize(
    "num_samples, fun, criterion",
    [
        (8, shapley, max_samples_criterion(2**10)),
        (10, permutation_shapley, max_samples_criterion(300)),
    ],
)
def test_shapley(analytic_shapley, fun: SemiValue, criterion: StoppingCriterion):
    u, exact_values = analytic_shapley
    values = fun(u, criterion)
    check_values(values, exact_values, rtol=0.1)


@pytest.mark.parametrize(
    "num_samples, fun, max_samples, criterion",
    [
        (8, shapley, 2**12, stderr_criterion(0.2, 1.0)),
        (10, permutation_shapley, 3000, stderr_criterion(0.1, 1.0)),
    ],
)
def test_shapley_convergence(
    analytic_shapley, fun: SemiValue, max_samples: int, criterion: StoppingCriterion
):
    u, exact_values = analytic_shapley
    values = fun(u, criterion)
    check_values(values, exact_values, rtol=0.1)
    assert values.status == ValuationStatus.Converged


@pytest.mark.parametrize("num_samples", [10])
@pytest.mark.parametrize(
    "fun, criterion, max_samples",
    [
        (beta_shapley, stderr_criterion(0.1, 1.0), 1000),
        (beta_shapley_paper, stderr_criterion(0.1, 1.0), 1000),
    ],
)
def test_beta_shapley(
    analytic_shapley, fun: SemiValue, criterion: StoppingCriterion, max_samples: int
):
    u, exact_values = analytic_shapley
    values = fun(u, criterion, alpha=1, beta=1)
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
def test_variance_criterion(
    values: "NDArray[np.float_]",
    variances: "NDArray[np.float_]",
    counts: "NDArray[np.int]",
    eps: float,
    values_ratio: float,
    status: ValuationStatus,
):
    assert (
        stderr_criterion(eps, values_ratio)(
            0,  # unused
            np.array(values, dtype=float),
            np.array(variances, dtype=float),
            np.array(counts, dtype=int),
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
    ],
)
def test_variance_criterion_exceptions(
    values: "NDArray[np.float_]",
    variances: "NDArray[np.float_]",
    counts: "NDArray[np.int]",
    exception,
):
    with pytest.raises(exception):
        stderr_criterion(0.1, 1.0)(0, values, variances, counts)


@pytest.mark.parametrize("n", [0, 10, 100])
@pytest.mark.parametrize(
    "coefficient",
    [
        beta_coefficient(1, 1),
        beta_coefficient(1, 16),
        banzhaf_coefficient,
        combinatorial_coefficient,
    ],
)
def test_coefficients(n: int, coefficient: Coefficient):
    r"""Coefficients for semi-values must fulfill:
    $$ \sum_{i=1}^{n}\choose{n-1}{j-1}w^{(n)}(j) = n $$
    """
    s = [math.comb(n - 1, j - 1) * coefficient(n, j - 1) for j in range(1, n + 1)]
    assert np.isclose(n, np.sum(s))

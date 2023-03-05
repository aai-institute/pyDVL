import math

import numpy as np
import pytest
from numpy.typing import NDArray

from pydvl.value import ValuationResult
from value import check_values

from pydvl.utils.status import Status
from pydvl.value.semivalues import (
    SemiValue,
    SVCoefficient,
    banzhaf_coefficient,
    beta_coefficient,
    beta_shapley,
    beta_shapley_paper,
    combinatorial_coefficient,
    permutation_shapley,
    shapley,
)
from pydvl.value.stopping import StoppingCriterion, MaxUpdates, StandardErrorRatio


@pytest.mark.parametrize(
    "num_samples, fun, criterion",
    [
        #
        (5, shapley, StandardErrorRatio(0.1, 1.0) & ~MaxUpdates(2**10)),
        (10, permutation_shapley, StandardErrorRatio(0.1, 1.0) & ~MaxUpdates(300)),
    ],
)
def test_shapley(analytic_shapley, fun: SemiValue, criterion: StoppingCriterion):
    u, exact_values = analytic_shapley
    values = fun(u, criterion)
    assert values.status == Status.Converged
    check_values(values, exact_values, rtol=0.1)


@pytest.mark.parametrize(
    "num_samples, fun, criterion",
    [
        # Uniform sampling with replacement is just too bad
        # (5, shapley, StandardErrorRatio(0.2, 1.0) | MaxUpdates(2**12),
        (10, permutation_shapley, StandardErrorRatio(0.1, 1.0) | MaxUpdates(600))
    ],
)
def test_shapley_convergence(
    analytic_shapley, fun: SemiValue, criterion: StoppingCriterion
):
    u, exact_values = analytic_shapley
    values = fun(u, criterion)
    check_values(values, exact_values, rtol=0.1)
    assert values.status == Status.Converged


@pytest.mark.parametrize("num_samples", [10])
@pytest.mark.parametrize(
    "fun, criterion",
    [
        (beta_shapley, StandardErrorRatio(0.1, 1.0) | MaxUpdates(100)),
        # (beta_shapley, finite_difference_criterion(7, 10, 0.05, 1) | MaxUpdates(100),
        (beta_shapley_paper, StandardErrorRatio(0.1, 1.0) | MaxUpdates(100)),
    ],
)
def test_beta_shapley(analytic_shapley, fun: SemiValue, criterion: StoppingCriterion):
    u, exact_values = analytic_shapley
    values = fun(u, criterion, alpha=1, beta=1)
    assert values.status == Status.Converged
    check_values(values, exact_values, rtol=0.1)


@pytest.mark.parametrize(
    "values, variances, counts, eps, values_ratio, status",
    [
        ([1, 1, 1], [0, 0, 0], [1, 1, 1], 1e-4, 1.0, Status.Converged),
        ([1, 2, 3], [0.01, 0.03, 0.08], [1, 1, 1], 0.1, 1.0, Status.Converged),
        ([1, 2, 3], [0.01, 0.03, 0.8], [1, 1, 1], 0.1, 1.0, Status.Pending),
        ([1, 2, 3], [0.01, 0.03, 0.8], [1, 1, 1], 0.1, 2 / 3, Status.Converged),
    ],
)
def test_stderr_criterion(
    values: NDArray[np.float_],
    variances: NDArray[np.float_],
    counts: NDArray[np.int_],
    eps: float,
    values_ratio: float,
    status: Status,
):
    v = ValuationResult(
        values=np.array(values, dtype=float),
        variances=np.array(variances, dtype=float),
        counts=np.array(counts, dtype=int),
    )
    assert StandardErrorRatio(eps, values_ratio)(v) == status


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
    values: NDArray[np.float_],
    variances: NDArray[np.float_],
    counts: NDArray[np.int_],
    exception,
):
    v = ValuationResult(
        values=np.array(values, dtype=float),
        variances=np.array(variances, dtype=float),
        counts=np.array(counts, dtype=int),
    )
    with pytest.raises(exception):
        StandardErrorRatio(0.1, 1.0)(v)


@pytest.mark.parametrize("n", [0, 10, 100])
@pytest.mark.parametrize(
    "coefficient",
    [
        beta_coefficient(1, 1),
        beta_coefficient(1, 16),
        beta_coefficient(4, 1),
        banzhaf_coefficient,
        combinatorial_coefficient,
    ],
)
def test_coefficients(n: int, coefficient: SVCoefficient):
    r"""Coefficients for semi-values must fulfill:
    $$ \sum_{i=1}^{n}\choose{n-1}{j-1}w^{(n)}(j) = n $$
    """
    s = [math.comb(n - 1, j - 1) * coefficient(n, j - 1) for j in range(1, n + 1)]
    assert np.isclose(n, np.sum(s))

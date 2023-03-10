import math

import numpy as np
import pytest
from value import check_values

from pydvl.utils.status import Status
from pydvl.value.semivalues import (
    SemiValue,
    SVCoefficient,
    banzhaf_coefficient,
    banzhaf_index,
    beta_coefficient,
    beta_shapley,
    beta_shapley_paper,
    shapley_coefficient,
    permutation_shapley,
    shapley,
)
from pydvl.value.stopping import AbsoluteStandardError, MaxUpdates, StoppingCriterion


@pytest.mark.parametrize(
    "num_samples, fun, criterion",
    [
        (5, shapley, AbsoluteStandardError(0.02, 1.0) | MaxUpdates(2**10)),
        (10, permutation_shapley, AbsoluteStandardError(0.02, 1.0) | MaxUpdates(300)),
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
        (10, permutation_shapley, AbsoluteStandardError(0.05, 1.0) | MaxUpdates(600))
    ],
)
def test_shapley_convergence(
    analytic_shapley, fun: SemiValue, criterion: StoppingCriterion
):
    u, exact_values = analytic_shapley
    values = fun(u, criterion)
    check_values(values, exact_values, rtol=0.1)
    assert values.status == Status.Converged


@pytest.mark.parametrize(
    "num_samples, fun, criterion",
    [
        (6, beta_shapley, AbsoluteStandardError(0.02, 1.0) | MaxUpdates(100)),
        # (beta_shapley, FiniteDifference(7, 10, 0.05, 1) | MaxUpdates(100),
        (6, beta_shapley_paper, AbsoluteStandardError(0.02, 1.0) | MaxUpdates(100)),
    ],
)
def test_beta_shapley(analytic_shapley, fun: SemiValue, criterion: StoppingCriterion):
    u, exact_values = analytic_shapley
    values = fun(u, criterion, alpha=1, beta=1)
    assert values.status == Status.Converged
    check_values(values, exact_values, rtol=0.1)


@pytest.mark.parametrize("num_samples", [5])
def test_banzhaf(analytic_banzhaf, num_samples):
    u, exact_values = analytic_banzhaf
    values = banzhaf_index(u, AbsoluteStandardError(0.02, 1.0) | MaxUpdates(2**10))
    check_values(values, exact_values, rtol=0.1)


@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize(
    "coefficient",
    [
        beta_coefficient(1, 1),
        beta_coefficient(1, 16),
        beta_coefficient(4, 1),
        banzhaf_coefficient,
        shapley_coefficient,
    ],
)
def test_coefficients(n: int, coefficient: SVCoefficient):
    r"""Coefficients for semi-values must fulfill:

    $$ \sum_{i=1}^{n}\choose{n-1}{j-1}w^{(n)}(j) = 1 $$

    Note that we depart from the usual definitions by including the factor $1/n$
    in the shapley and beta coefficients.
    """
    s = [math.comb(n - 1, j - 1) * coefficient(n, j - 1) for j in range(1, n + 1)]
    assert np.isclose(1, np.sum(s))

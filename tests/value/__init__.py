from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

from pydvl.utils import Utility
from pydvl.value.result import ValuationResult


def polynomial(coefficients, x):
    powers = np.arange(len(coefficients))
    return np.power(x, np.tile(powers, (len(x), 1)).T).T @ coefficients


def check_total_value(
    u: Utility, values: ValuationResult, rtol: float = 0.05, atol: float = 1e-6
):
    """Checks absolute distance between total and added values.
    Shapley value is supposed to fulfill the total value axiom."""
    total_utility = u(u.data.indices)
    # We can use relative tolerances if we don't have the range of the scorer.
    np.testing.assert_allclose(
        np.sum(values.values), total_utility, rtol=rtol, atol=atol
    )


def check_exact(
    values: ValuationResult,
    exact_values: ValuationResult,
    rtol: float = 0.1,
    atol: float = 1e-6,
):
    """Compares ranks and values."""

    values.sort()
    exact_values.sort()

    np.testing.assert_equal(values.indices, exact_values.indices, "Ranks do not match")
    np.testing.assert_allclose(
        values.values,
        exact_values.values,
        rtol=rtol,
        atol=atol,
        err_msg="Values do not match",
    )


def check_values(
    values: ValuationResult,
    exact_values: ValuationResult,
    rtol: float = 0.1,
    atol: float = 1e-5,
    *,
    extra_values_names: Sequence[str] = tuple(),
):
    """Compares values.

    Asserts that `|value - exact_value| < |exact_value| * rtol + atol` for
    all pairs of `value`, `exact_value` with equal index.

    :param values:
    :param exact_values:
    :param rtol: relative tolerance of elements in `values` with respect to
        elements in `exact_values`. E.g. if rtol = 0.1, and atol = 0 we must
        have |value - exact_value|/|exact_value| < 0.1 for every value
    :param atol: absolute tolerance of elements in `values` with respect to
        elements in `exact_values`. E.g. if atol = 0.1, and rtol = 0 we must
        have |value - exact_value| < 0.1 for every value.
    :param extra_values_names: Sequence of names of extra values that should
        also be compared.
    """
    values.sort()
    exact_values.sort()

    np.testing.assert_allclose(values.values, exact_values.values, rtol=rtol, atol=atol)
    for name in extra_values_names:
        np.testing.assert_allclose(
            getattr(values, name), getattr(exact_values, name), rtol=rtol, atol=atol
        )


def check_rank_correlation(
    values: ValuationResult,
    exact_values: ValuationResult,
    k: int = None,
    threshold: float = 0.9,
):
    """Checks that the indices of `values` and `exact_values` follow the same
    order (by value), with some slack, using Spearman's correlation.

    Runs an assertion for testing.

    :param values: The values and indices to test
    :param exact_values: The ground truth
    :param k: Consider only these many, starting from the top.
    :param threshold: minimal value for spearman correlation for the test to
        succeed
    """
    # FIXME: estimate proper threshold for spearman

    k = k or len(values)

    values.sort()
    exact_values.sort()

    top_k = np.array([it.index for it in values[-k:]])
    top_k_exact = np.array([it.index for it in exact_values[-k:]])

    correlation, pvalue = spearmanr(top_k, top_k_exact)
    assert correlation >= threshold, f"{correlation} < {threshold}"

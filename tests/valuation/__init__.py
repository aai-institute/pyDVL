from __future__ import annotations

from typing import Sequence, Type, TypeVar

import numpy as np
from scipy.stats import spearmanr

from pydvl.valuation.result import ValuationResult
from pydvl.valuation.types import Sample
from pydvl.valuation.utility.base import UtilityBase


def polynomial(coefficients, x):
    powers = np.arange(len(coefficients))
    return np.power(x, np.tile(powers, (len(x), 1)).T).T @ coefficients


def check_total_value(
    u: UtilityBase, values: ValuationResult, rtol: float = 0.05, atol: float = 1e-6
):
    """Checks absolute distance between total and added values.
    Shapley value is supposed to fulfill the total value axiom."""
    assert u.training_data is not None
    total_utility = u(Sample(idx=None, subset=u.training_data.indices))
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

    values = values.sort()
    exact_values = exact_values.sort()

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
    values = values.sort()
    exact_values = exact_values.sort()

    np.testing.assert_allclose(values.values, exact_values.values, rtol=rtol, atol=atol)
    for name in extra_values_names:
        np.testing.assert_allclose(
            getattr(values, name), getattr(exact_values, name), rtol=rtol, atol=atol
        )


def check_rank_correlation(
    values: ValuationResult,
    exact_values: ValuationResult,
    k: int | None = None,
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

    values = values.sort()
    exact_values = exact_values.sort()

    top_k = np.array([it.idx for it in values[-k:]])
    top_k_exact = np.array([it.idx for it in exact_values[-k:]])

    correlation, pvalue = spearmanr(top_k, top_k_exact)
    assert correlation >= threshold, f"{correlation} < {threshold}"


def is_lambda(obj) -> bool:
    return isinstance(obj, type(lambda: None)) and obj.__name__ == "<lambda>"


T = TypeVar("T")


def recursive_make(t: Type[T], t_kwargs: dict, **lambda_args) -> T:
    """Recursively instantiate classes with arguments.

    If a value in `t_kwargs` is a tuple, it is assumed to be a class and its
    arguments. If a value is a callable, it is called with the argument in
    `lambda_args` of the same name as the key. lambdas may only accept one argument.
    Arguments are not "exhausted" in any way and may be reused.
    """
    t_kwargs = t_kwargs.copy()  # careful with mutable inputs...
    for k, v in t_kwargs.items():
        if is_lambda(v):
            if k in lambda_args:
                t_kwargs[k] = v(lambda_args[k])
            else:
                t_kwargs[k] = v()
        elif isinstance(v, tuple) and isinstance(v[0], type):
            t_kwargs[k] = recursive_make(*v, **lambda_args)
    return t(**t_kwargs)

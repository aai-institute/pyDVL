import numpy as np
import pytest

from collections import OrderedDict
from conftest import TolerateErrors
from functools import partial
from valuation.shapley import combinatorial_montecarlo_shapley, \
    permutation_montecarlo_shapley, truncated_montecarlo_shapley,\
    permutation_exact_shapley
from valuation.utils import Dataset, SupervisedModel
from valuation.utils.numeric import lower_bound_hoeffding, spearman, utility
from valuation.utils.parallel import run_and_gather, available_cpus
from valuation.utils.types import Scorer


def check_total_value(model: SupervisedModel,
                      data: Dataset,
                      values: OrderedDict,
                      scoring: Scorer = None,
                      rtol: float = 0.01):
    """ Checks absolute distance between total and added values.
     Shapley value is supposed to fulfill the total value axiom."""
    total_utility = utility(model, data, frozenset(data.indices), scoring)
    values = np.array(list(values.values()))
    # We use relative tolerances here because we don't have the range of the
    # scorer.
    assert np.isclose(values.sum(), total_utility, rtol=rtol)


def check_exact(values: OrderedDict, exact_values: OrderedDict, eps: float):
    """ Compares ranks and values. """

    k = list(values.keys())
    ek = list(exact_values.keys())

    assert np.all(k == ek)

    v = np.array(list(values.values()))
    ev = np.array(list(exact_values.values()))

    assert np.allclose(v, ev, atol=eps)


def check_rank_correlation(values: OrderedDict, exact_values: OrderedDict,
                           n: int = None, threshold: float = 0.9):
    # FIXME: estimate proper threshold for spearman
    if n is not None:
        raise NotImplementedError
    else:
        n = len(values)
    ranks = np.array(list(values.keys())[:n])
    ranks_exact = np.array(list(exact_values.keys())[:n])

    assert spearman(ranks, ranks_exact) >= threshold


# pedantic...
@pytest.mark.parametrize(
    "passes, values_a, values_b, eps",
    [(False,
      OrderedDict([(k, k+0.01) for k in range(10)]),
      OrderedDict([(k, k) for k in range(10)]),
      0.001),
     (True,
      OrderedDict([(k, k + 0.01) for k in range(10)]),
      OrderedDict([(k, k) for k in range(10)]),
      0.01),
     (True,
      OrderedDict([(k, k) for k in range(10)]),
      OrderedDict([(k, k) for k in range(10)]),
      0)
     ])
def test_compare(passes, values_a, values_b, eps):
    if passes:
        check_exact(values_a, values_b, eps)
    else:
        with pytest.raises(AssertionError):
            check_exact(values_a, values_b, eps)


@pytest.mark.parametrize(
        "scoring, rtol",
        [('r2', 0.01),
         ('neg_mean_squared_error', 0.01),
         ('neg_median_absolute_error', 0.01)])
def test_combinatorial_exact_shapley(exact_shapley, rtol):
    model, data, values, scoring = exact_shapley
    check_total_value(model, data, values, scoring, rtol=rtol)
    # TODO: compute "manually" for fixed values and check


@pytest.mark.parametrize(
    "scoring, rtol, eps",
    [('r2', 0.01, 0.01),
     ('neg_mean_squared_error', 0.01, 0.01),
     ('neg_median_absolute_error', 0.01, 0.01)])
def test_permutation_exact_shapley(scoring, rtol, eps, exact_shapley):
    model, data, exact_values, scoring = exact_shapley
    values_p = permutation_exact_shapley(model, data, scoring=scoring, progress=False)
    check_total_value(model, data, values_p, scoring, rtol=rtol)
    check_exact(values_p, exact_values, eps=eps)


# FIXME: this is not deterministic
@pytest.mark.parametrize(
    "fun, scoring, score_range, delta, eps",
    [(permutation_montecarlo_shapley,
      'neg_mean_squared_error', 2, 0.01, 0.04),
     (combinatorial_montecarlo_shapley,
      'neg_mean_squared_error', 2, 0.01, 0.04)])
def test_montecarlo_shapley(fun, scoring, score_range, delta, eps, exact_shapley):
    model, data, exact_values, scoring = exact_shapley
    num_cpus = min(available_cpus(), len(data))
    num_runs = 10

    # FIXME: The lower bound does not apply to all methods
    # Sample bound: |value - estimate| < eps holds with probability 1-𝛿
    max_iterations = lower_bound_hoeffding(
            delta=delta, eps=eps, score_range=score_range)

    print(f"test_montecarlo_shapley running for {max_iterations} iterations")

    _fun = partial(fun, model=model, data=data, scoring=scoring,
                   max_iterations=max_iterations, progress=False)
    results = run_and_gather(_fun, num_jobs=num_cpus, num_runs=num_runs)

    delta_errors = TolerateErrors(int(delta*len(results)), AssertionError)
    for values, _ in results:
        with delta_errors:
            # FIXME: test for total value never passes! (completely off)
            # Trivial bound on total error using triangle inequality
            # check_total_value(model, data, values, rtol=len(data)*eps)
            check_rank_correlation(values, exact_values, threshold=0.9)


# FIXME: this is not deterministic
@pytest.mark.parametrize(
        "scoring, score_range",
        [('neg_mean_squared_error', 2),
         ('r2', 2)])
def test_truncated_montecarlo_shapley(scoring, score_range, exact_shapley):
    model, data, exact_values, scoring = exact_shapley
    num_cpus = min(available_cpus(), len(data))
    num_runs = 10
    delta = 0.01  # Sample bound holds with probability 1-𝛿
    eps = 0.04

    min_permutations =\
        lower_bound_hoeffding(delta=delta, eps=eps, score_range=score_range)

    print(f"test_truncated_montecarlo_shapley running for {num_runs} runs "
          f" of max. {min_permutations} iterations each")

    fun = partial(truncated_montecarlo_shapley, model=model, data=data,
                  scoring=scoring, bootstrap_iterations=10, min_scores=5,
                  score_tolerance=0.1, min_values=10, value_tolerance=eps,
                  max_iterations=min_permutations, num_workers=num_cpus,
                  progress=False)
    results = []
    for i in range(num_runs):
        results.append(fun(run_id=i))

    delta_errors = TolerateErrors(int(delta * len(results)), AssertionError)
    for values, _ in results:
        with delta_errors:
            # Trivial bound on total error using triangle inequality
            check_total_value(model, data, values, rtol=len(data)*eps)
            check_rank_correlation(values, exact_values, threshold=0.8)

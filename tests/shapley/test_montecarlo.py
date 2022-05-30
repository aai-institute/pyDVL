import numpy as np
import pytest
import logging

from tests.conftest import TolerateErrors, check_exact, check_rank_correlation, \
    check_total_value, check_values, polynomial, polynomial_dataset
from functools import partial
from valuation.shapley import combinatorial_montecarlo_shapley, \
    permutation_montecarlo_shapley, truncated_montecarlo_shapley
from valuation.utils.numeric import lower_bound_hoeffding
from valuation.utils.parallel import MapReduceJob, available_cpus, map_reduce
from valuation.utils import Utility
from sklearn.linear_model import LinearRegression
from valuation.shapley import combinatorial_exact_shapley

# noinspection PyTestParametrized
@pytest.mark.parametrize(
    "num_samples, fun, perc_atol, max_iterations",
    [
    # (12, permutation_montecarlo_shapley, 10, 1),
     (7, combinatorial_montecarlo_shapley, 10, 1e+3),
    ])
def test_analytic_montecarlo_shapley(analytic_shapley, fun, perc_atol, max_iterations):
    u, exact_values = analytic_shapley
    num_jobs = min(8, available_cpus())

    from valuation.utils.logging import start_logging_server
    start_logging_server()

    logging.info(f"Using max iterations {max_iterations}")

    values, _ = fun( u, max_iterations=max_iterations, progress=False, num_jobs=num_jobs)

    logging.info(f"These are the values: {values}")
    logging.info(f"These are the exact values: {exact_values}")
    check_values(values, exact_values, perc_atol=perc_atol)

@pytest.mark.parametrize(
    "a, b, fun, score_type, perc_atol, max_iterations",
    [
    (2, 1, permutation_montecarlo_shapley, "r2", 50, 1000),
    (2, 0, permutation_montecarlo_shapley, "explained_variance", 50, 1000),
    (2, 2, permutation_montecarlo_shapley,  "neg_median_absolute_error", 50, 1000),
    #  (12, combinatorial_montecarlo_shapley, 1e-2, 1e-2),
    ])
def test_linear_montecarlo_shapley(linear_dataset, fun, score_type, perc_atol, max_iterations):
    num_jobs = min(8, available_cpus())

    from valuation.utils.logging import start_logging_server
    start_logging_server()
    linear_utility = Utility(LinearRegression(), data=linear_dataset, scoring=score_type)

    values, _ = fun(linear_utility, max_iterations=max_iterations, progress=False, num_jobs=num_jobs)
    exact_values = combinatorial_exact_shapley(linear_utility, progress=False)
    logging.info(f"These are the values: {values}")
    logging.info(f"These are the exact values: {exact_values}")
    check_values(values, exact_values, perc_atol=perc_atol)


# noinspection PyTestParametrized
@pytest.mark.skip
@pytest.mark.parametrize("num_samples", [200])
def test_truncated_montecarlo_shapley(analytic_shapley):
    u, exact_values = analytic_shapley
    num_cpus = min(available_cpus(), len(u.data))
    num_runs = 10
    delta = 0.01  # Sample bound holds with probability 1-𝛿
    eps = 0.05

    min_permutations =\
        lower_bound_hoeffding(delta=delta, eps=eps, score_range=1)

    print(f"test_truncated_montecarlo_shapley running for {num_runs} runs "
          f" of max. {min_permutations} iterations each")

    fun = partial(truncated_montecarlo_shapley, u=u, bootstrap_iterations=10,
                  min_scores=5, score_tolerance=0.1, min_values=10,
                  value_tolerance=eps, max_iterations=min_permutations,
                  num_workers=num_cpus, progress=False)
    results = []
    for i in range(num_runs):
        results.append(fun(run_id=i))

    delta_errors = TolerateErrors(max(1, int(delta * len(results))))
    for values, _ in results:
        with delta_errors:
            # Trivial bound on total error using triangle inequality
            check_total_value(u, values, atol=len(u.data)*eps)
            check_rank_correlation(values, exact_values, threshold=0.8)
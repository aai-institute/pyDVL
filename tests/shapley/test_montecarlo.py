import numpy as np
import pytest
import logging

from tests.conftest import TolerateErrors, check_exact, check_rank_correlation, \
    check_total_value, polynomial, polynomial_dataset
from functools import partial
from valuation.shapley import combinatorial_montecarlo_shapley, \
    permutation_montecarlo_shapley, truncated_montecarlo_shapley
from valuation.utils.numeric import lower_bound_hoeffding
from valuation.utils.parallel import MapReduceJob, available_cpus, map_reduce


# noinspection PyTestParametrized
@pytest.mark.parametrize(
    "num_samples, delta, eps",
    [(12, 1e-2, 1e-2),
    ])
def test_analytic_permutation_montecarlo(analytic_shapley, delta, eps):
    u, exact_values = analytic_shapley

    # Sample bound: |value - estimate| < ε holds with probability 1-𝛿
    hoeff_iterations = lower_bound_hoeffding(delta=delta, eps=eps,
                                            score_range=1)

    values = permutation_montecarlo_shapley(u, max_iterations=hoeff_iterations,
                   progress=False, num_jobs=n_)
    logging.info(f"This is results: {len(results)}")

    for values in results:
            # Trivial bound on total error using triangle inequality
            check_total_value(u, values, atol=len(u.data)*eps)
            check_exact(values, exact_values, atol=eps)
            check_rank_correlation(values, exact_values, threshold=0.9)


# noinspection PyTestParametrized
# @pytest.mark.parametrize("num_samples", [200])
# def test_truncated_montecarlo_shapley(exact_shapley):
#     u, exact_values = exact_shapley
#     num_cpus = min(available_cpus(), len(u.data))
#     num_runs = 10
#     delta = 0.01  # Sample bound holds with probability 1-𝛿
#     eps = 0.05

#     min_permutations =\
#         lower_bound_hoeffding(delta=delta, eps=eps, score_range=1)

#     print(f"test_truncated_montecarlo_shapley running for {num_runs} runs "
#           f" of max. {min_permutations} iterations each")

#     fun = partial(truncated_montecarlo_shapley, u=u, bootstrap_iterations=10,
#                   min_scores=5, score_tolerance=0.1, min_values=10,
#                   value_tolerance=eps, max_iterations=min_permutations,
#                   num_workers=num_cpus, progress=False)
#     results = []
#     for i in range(num_runs):
#         results.append(fun(run_id=i))

#     delta_errors = TolerateErrors(max(1, int(delta * len(results))))
#     for values, _ in results:
#         with delta_errors:
#             # Trivial bound on total error using triangle inequality
#             check_total_value(u, values, atol=len(u.data)*eps)
#             check_rank_correlation(values, exact_values, threshold=0.8)
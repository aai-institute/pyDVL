import numpy as np
import pytest

from collections import OrderedDict
from tests.conftest import TolerateErrors, check_exact, check_rank_correlation, \
    check_total_value, polynomial, polynomial_dataset
from functools import partial
from valuation.shapley import combinatorial_exact_shapley, \
    combinatorial_montecarlo_shapley, \
    permutation_montecarlo_shapley, truncated_montecarlo_shapley, \
    permutation_exact_shapley
from valuation.utils.numeric import lower_bound_hoeffding
from valuation.utils.parallel import MapReduceJob, available_cpus, map_reduce


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
def test_check_exact(passes, values_a, values_b, eps):
    if passes:
        check_exact(values_a, values_b, eps)
    else:
        with pytest.raises(AssertionError):
            check_exact(values_a, values_b, eps)


# noinspection PyTestParametrized
@pytest.mark.parametrize(
        "num_samples, fun, value_atol, total_atol",
        [(12, combinatorial_exact_shapley, 1e-5, 1e-5),
         (6, permutation_exact_shapley, 1e-5, 1e-5)])
def test_exact_shapley(exact_shapley, fun, value_atol, total_atol):
    u, exact_values = exact_shapley
    values_p = fun(u, progress=False)
    check_total_value(u, values_p, atol=total_atol)
    check_exact(values_p, exact_values, atol=value_atol)


# noinspection PyTestParametrized
@pytest.mark.parametrize(
    "num_samples, fun, delta, eps",
    [(24, permutation_montecarlo_shapley, 1e-2, 1e-2),
     # FIXME: this does not work. At all
     # (100, combinatorial_montecarlo_shapley, 1e-2, 1e-2)
    ])
def test_montecarlo_shapley(exact_shapley, fun, delta, eps):
    u, exact_values = exact_shapley
    jobs_per_run = min(6, available_cpus(), len(u.data))
    num_runs = min(3, available_cpus() // jobs_per_run)

    from valuation.utils.logging import start_logging_server
    start_logging_server()

    max_iterations = None
    if fun == permutation_montecarlo_shapley:
        # Sample bound: |value - estimate| < ε holds with probability 1-𝛿
        max_iterations = lower_bound_hoeffding(delta=delta, eps=eps,
                                               score_range=1)
    elif fun == combinatorial_montecarlo_shapley:
        # FIXME: Abysmal performance if this is required
        max_iterations = 2**(len(u.data))

    print(f"test_montecarlo_shapley running for {max_iterations} iterations "
          f"per each of {jobs_per_run} jobs. Repeated {num_runs} times.")

    _fun = partial(fun, max_iterations=max_iterations // jobs_per_run,
                   progress=False, num_jobs=jobs_per_run)
    job = MapReduceJob.from_fun(_fun, lambda r: r[0][0])
    results = map_reduce(job, u, num_jobs=num_runs, num_runs=num_runs)

    delta_errors = TolerateErrors(max(1, int(delta*len(results))))
    for values in results:
        with delta_errors:
            # Trivial bound on total error using triangle inequality
            check_total_value(u, values, atol=len(u.data)*eps)
            check_rank_correlation(values, exact_values, threshold=0.9)


# noinspection PyTestParametrized
@pytest.mark.parametrize("num_samples", [200])
def test_truncated_montecarlo_shapley(exact_shapley):
    u, exact_values = exact_shapley
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

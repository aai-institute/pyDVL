import numpy as np

from collections import OrderedDict
from functools import partial
from valuation.shapley import combinatorial_montecarlo_shapley, \
    permutation_montecarlo_shapley, truncated_montecarlo_shapley,\
    combinatorial_exact_shapley, permutation_exact_shapley
from valuation.utils.numeric import lower_bound_hoeffding
from valuation.utils.parallel import parallel_wrap, run_and_gather, \
    available_cpus


def compare(values_a: OrderedDict, values_b: OrderedDict, eps: float):
    assert np.all(values_a.keys() == values_b.keys())
    assert np.allclose(np.array(list(values_a.values())),
                       np.array(list(values_a.values())), atol=eps)


def test_combinatorial_exact_shapley():
    # TODO: compute "manually" for fixed values and check
    pass


def test_permutation_exact_shapley(exact_shapley):
    model, data, exact_values = exact_shapley
    values_p = permutation_exact_shapley(model, data, progress=False)
    compare(values_p, exact_values, eps=0.01)


def test_permutation_montecarlo_shapley(exact_shapley):
    model, data, exact_values = exact_shapley
    num_cpus = min(available_cpus(), len(data))
    num_runs = 1  # TODO: average over multiple runs?
    eps = 0.02
    score_range = 1  # FIXME: bogus (R^2 is unbounded below)

    # FIXME: this is non-deterministic
    min_permutations = lower_bound_hoeffding(delta=0.01, eps=eps, r=score_range)
    print(f"test_naive_montecarlo_shapley running for {min_permutations} "
          f"iterations")
    fun = partial(permutation_montecarlo_shapley, model, data,
                  max_permutations=min_permutations, progress=False)
    wrapped = parallel_wrap(fun, ("indices", data.ilocs), num_jobs=num_cpus)
    values_m, _ = run_and_gather(wrapped, num_runs=num_runs, progress=False)
    compare(values_m[0], exact_values, eps)


def test_combinatorial_montecarlo_shapley(exact_shapley):
    model, data, exact_values = exact_shapley
    num_cpus = min(available_cpus(), len(data))
    num_runs = 1  # TODO: average over multiple runs?
    eps = 0.02

    fun = partial(combinatorial_montecarlo_shapley, model, data, progress=False)
    wrapped = parallel_wrap(fun, ("indices", data.ilocs), num_jobs=num_cpus)
    values_cm, _ = run_and_gather(wrapped, num_runs=num_runs, progress=False)
    compare(values_cm[0], exact_values, eps)


def test_truncated_montecarlo_shapley(exact_shapley):
    model, data, exact_values = exact_shapley
    num_cpus = min(available_cpus(), len(data))
    num_runs = 1  # TODO: average over multiple runs?
    eps = 0.02
    score_range = 1  # FIXME: bogus (R^2 is unbounded below)

    # FIXME: this is non-deterministic
    min_permutations = lower_bound_hoeffding(delta=0.01, eps=eps, r=score_range)
    print(f"test_truncated_montecarlo_shapley running for {num_runs} runs "
          f" of max. {min_permutations} iterations each")

    wrapped = partial(truncated_montecarlo_shapley,
                      model=model, data=data, bootstrap_iterations=10,
                      min_samples=5, score_tolerance=1e-1, min_values=10,
                      value_tolerance=eps, max_permutations=min_permutations,
                      num_workers=num_cpus, progress=False)
    values_tm, _ = run_and_gather(wrapped, num_runs=num_runs, progress=False)
    values_tm = OrderedDict(((k, vv[-1]) for k, vv in values_tm[0].items()))

    compare(values_tm, exact_values, eps)

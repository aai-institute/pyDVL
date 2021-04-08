import numpy as np

from functools import partial
from valuation.shapley import combinatorial_montecarlo_shapley, \
    permutation_montecarlo_shapley, truncated_montecarlo_shapley,\
    combinatorial_exact_shapley, permutation_exact_shapley
from valuation.utils.numeric import lower_bound_hoeffding
from valuation.utils.parallel import parallel_wrap, run_and_gather, \
    available_cpus


def test_combinatorial_exact_shapley():
    # TODO: compute "manually" for fixed values and check
    pass


def test_permutation_exact_shapley(exact_shapley):
    model, data, values = exact_shapley
    values_p = permutation_exact_shapley(model, data, progress=False)

    assert np.all(values_p.keys() == values.keys())
    assert np.allclose(np.array(list(values_p.values())), values, rtol=1e-1)


def test_permutation_montecarlo_shapley(exact_shapley):
    model, data, values = exact_shapley

    num_cpus = min(available_cpus(), len(data))

    # FIXME: this is non-deterministic
    # FIXME: the range is bogus (R^2 is unbounded below)
    max_iterations = lower_bound_hoeffding(delta=0.01, eps=0.01, r=1)
    print(f"test_naive_montecarlo_shapley running for {max_iterations} iterations")
    fun = partial(permutation_montecarlo_shapley, model, data,
                  max_iterations=max_iterations, progress=False)
    wrapped = parallel_wrap(fun, ("indices", data.ilocs),
                            num_jobs=num_cpus)
    # TODO: average over multiple runs
    values_m, _ = run_and_gather(wrapped, num_runs=1, progress=False)
    values_m = values_m[0]

    assert np.all(values_m.keys() == values.keys())
    assert np.allclose(np.array(list(values_m.values())), values, rtol=1e-1)


def test_combinatorial_montecarlo_shapley(exact_shapley):
    model, data, values = exact_shapley
    num_cpus = min(available_cpus(), len(data))

    fun = partial(combinatorial_montecarlo_shapley, model, data,
                  progress=False)
    wrapped = parallel_wrap(fun, ("indices", data.ilocs),
                            num_jobs=num_cpus)
    values_m, _ = run_and_gather(wrapped, num_runs=1, progress=False)
    values_m = values_m[0]

    assert np.all(values_m.keys() == values.keys())
    assert np.allclose(np.array(list(values_m.values())), values, rtol=1e-1)


def test_truncated_montecarlo_shapley(exact_shapley):
    model, data, values = exact_shapley
    num_cpus = min(available_cpus(), len(data))
    num_runs = 1  # TODO: average over multiple runs

    # FIXME: this is non-deterministic
    # FIXME: the range is bogus (R^2 is unbounded below)
    max_iterations = lower_bound_hoeffding(delta=0.01, eps=0.01, r=1)
    print(f"test_truncated_montecarlo_shapley running for {num_runs} runs "
          f" of max. {max_iterations} iterations each")

    wrapped = partial(truncated_montecarlo_shapley, model=model,
                      data=data, bootstrap_iterations=10,
                      min_samples=5, score_tolerance=1e-1, min_values=10,
                      value_tolerance=1e-3, max_permutations=max_iterations,
                      num_workers=num_cpus, worker_progress=False)
    values_m, _ = run_and_gather(wrapped, num_runs=num_runs, progress=False)
    values_m = values_m[0]

    assert np.all(values_m.keys() == values.keys())
    assert np.allclose(np.array(list(values_m.values()))[:, -1], values,
                       rtol=1e-1)

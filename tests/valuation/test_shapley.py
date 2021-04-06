import numpy as np

from functools import partial
from sklearn.linear_model import LinearRegression
from valuation.shapley.montecarlo import naive_montecarlo_shapley
from valuation.shapley.naive import exact_combinatorial_shapley, \
    exact_permutation_shapley, montecarlo_combinatorial_shapley
from valuation.utils.numeric import lower_bound_hoeffding
from valuation.utils.parallel import parallel_wrap, run_and_gather, \
    available_cpus


def test_exact_naive_shapley(linear_dataset):
    model = LinearRegression()
    values_p = exact_permutation_shapley(model, linear_dataset, progress=False)
    values_c = exact_combinatorial_shapley(model, linear_dataset, progress=False)

    assert np.alltrue(values_p.keys() == values_c.keys())
    assert np.allclose(np.array(list(values_p.values())),
                       np.array(list(values_c.values())),
                       atol=1e-6)


def test_naive_montecarlo_shapley(linear_dataset):
    num_cpus = min(available_cpus(), len(linear_dataset))
    model = LinearRegression()

    # FIXME: this is non-deterministic
    # FIXME: the range is bogus (R^2 is unbounded below)
    max_iterations = lower_bound_hoeffding(delta=0.01, eps=0.01, r=1)
    print(f"test_naive_montecarlo_shapley running for {max_iterations} iterations")
    fun = partial(naive_montecarlo_shapley, model, linear_dataset,
                  max_iterations=max_iterations, progress=False)
    wrapped = parallel_wrap(fun, ("indices", linear_dataset.ilocs),
                            num_jobs=num_cpus)
    # TODO: average over multiple runs
    values_m, _ = run_and_gather(wrapped, num_runs=1, progress=False)
    values_m = values_m[0]
    values_c = exact_combinatorial_shapley(model, linear_dataset,
                                           progress=False)

    assert np.alltrue(values_m.keys() == values_c.keys())
    assert np.allclose(np.array(list(values_m.values())),
                       np.array(list(values_c.values())),
                       rtol=1e-1)


def test_montecarlo_combinatorial_shapley(linear_dataset):
    num_cpus = min(available_cpus(), len(linear_dataset))
    model = LinearRegression()

    fun = partial(montecarlo_combinatorial_shapley, model, linear_dataset,
                  progress=False)
    wrapped = parallel_wrap(fun, ("indices", linear_dataset.ilocs),
                            num_jobs=num_cpus)
    values_m, _ = run_and_gather(wrapped, num_runs=1, progress=False)
    values_m = values_m[0]
    values_c = exact_combinatorial_shapley(model, linear_dataset,
                                           progress=False)

    assert np.alltrue(values_m.keys() == values_c.keys())
    assert np.allclose(np.array(list(values_m.values())),
                       np.array(list(values_c.values())),
                       atol=1e-1)

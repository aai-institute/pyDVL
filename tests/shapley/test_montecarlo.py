import numpy as np
import pytest

from tests.conftest import TolerateErrors, check_exact, check_rank_correlation, \
    check_total_value, polynomial, polynomial_dataset
from functools import partial
from valuation.shapley import combinatorial_montecarlo_shapley, \
    permutation_montecarlo_shapley, truncated_montecarlo_shapley
from valuation.utils.numeric import lower_bound_hoeffding
from valuation.utils.parallel import MapReduceJob, available_cpus, map_reduce


# noinspection PyTestParametrized
@pytest.mark.parametrize(
    "num_samples, fun, delta, eps",
    [(24, permutation_montecarlo_shapley, 1e-2, 1e-2),
     # FIXME: this does not work. At all
     # (100, combinatorial_montecarlo_shapley, 1e-2, 1e-2)
    ])
def test_montecarlo_shapley(analytic_shapley, fun, delta, eps):
    u, exact_values = analytic_shapley
    jobs_per_run = min(6, available_cpus(), len(u.data))
    num_runs = min(3, available_cpus() // jobs_per_run)

    # from valuation.utils.logging import start_logging_server
    # start_logging_server()

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


# noinspection PyTestParametrized
@pytest.mark.parametrize(
        "coefficients",
        [(np.random.randint(-3, 3, size=5))]
        )
def test_removal(polynomial_dataset):
    # Poly of degree 4 => 5 points enough for interpolation
    # => coalitions of > 5 points shouldn't increase value over 5 points

    d, coeffs = polynomial_dataset  # Renaming for convenience
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LinearRegression
    model = make_pipeline(PolynomialFeatures(len(coeffs)-1), LinearRegression())

    n = len(d)
    model.fit(d.x_train, d.y_train)
    predicted = [model.predict(d.x_test)]

    x_cont = d.x_train.reshape(-1,) + np.random.uniform(-0.05, 0.05, size=len(d))
    x_cont = x_cont[::2]
    y_cont = polynomial(np.random.normal(loc=coeffs, scale=0.3), x_cont)
    xtrain = np.concatenate([d.x_train, x_cont.reshape(-1, 1)], axis=0)
    ytrain = np.concatenate([d.y_train, y_cont.reshape(-1,)], axis=0)
    for i in range(len(d), len(xtrain)):
        model.fit(xtrain[:i+1], ytrain[:i+1])
        ypred = model.predict(d.x_test)
        predicted.append(ypred)

    test_indices = np.argsort(d.x_test, axis=0).reshape(-1, )
    xx = np.arange(-1, 1, 0.1)
    yy = polynomial(coeffs, xx)

    from matplotlib import pyplot as plt
    for i, ypred in enumerate(predicted):
        plt.figure(dpi=300)
        plt.scatter(d.x_train[:n], d.y_train[:n], label="Training (in-dist)")
        plt.scatter(d.x_test, d.y_test, label="Test")

        if i > 0:
            plt.scatter(x_cont[:i], y_cont[:i], label="Training (out of dist)")
        plt.plot(xx, yy, label="True")
        plt.plot(d.x_test[test_indices], ypred[test_indices], label="Predicted")
        plt.ylim(min(d.y_train[:n].min(), y_cont.min()) - 1,
                 max(d.y_train[:n].max(), y_cont.max()) + 1)
        plt.legend()
        plt.title(d.description)
        plt.show()

    d.x_train = xtrain
    d.y_train = ytrain

    from valuation.shapley import combinatorial_exact_shapley
    from valuation.utils import Utility
    u = Utility(model, d, "neg_median_absolute_error")
    values = combinatorial_exact_shapley(u, progress=False)

    print("******")
    print(values)
    print("******")
    high_to_low = list(reversed(values))
    print(high_to_low)
    print("******")
    print(list(np.round(d.x_train[high_to_low].reshape(-1,), 2)))
    print(list(np.round(d.y_train[high_to_low], 2)))

    take = 5
    plt.figure(dpi=300)
    plt.scatter(d.x_train[:n], d.y_train[:n], label="Training (in-dist)")
    plt.scatter(d.x_test, d.y_test, label="Test")
    plt.scatter(d.x_train[high_to_low][:take], d.y_train[high_to_low][:take],
                marker='x', label='High value')
    plt.scatter(x_cont, y_cont, label="Training (out of dist)")
    plt.plot(xx, yy, label="True")
    plt.plot(d.x_test[test_indices], predicted[-1][test_indices], label="Predicted")

    model.fit(d.x_train[high_to_low][:take], d.y_train[high_to_low][:take])
    ypred = model.predict(d.x_test)
    plt.plot(d.x_test, ypred, label='High: prediction')
    plt.title(d.description)

    plt.legend()
    plt.show()
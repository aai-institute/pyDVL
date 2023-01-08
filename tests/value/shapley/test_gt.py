import pytest

from pydvl.value.shapley.gt import group_testing_shapley

from .. import check_values


@pytest.mark.parametrize(
    "num_samples, fun, rtol, max_iterations, kwargs",
    [
        (10, group_testing_shapley, 0.1, 1000, {}),
    ],
)
def test_group_testing_shapley(
    num_samples, analytic_shapley, fun, rtol, max_iterations, kwargs
):
    u, exact_values = analytic_shapley

    values = fun(
        u, max_iterations=int(max_iterations), progress=False, n_jobs=1, **kwargs
    )

    check_values(values, exact_values, rtol=rtol)

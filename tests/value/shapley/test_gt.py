import pytest

from pydvl.value.shapley.gt import group_testing_shapley

from .. import check_values


@pytest.mark.parametrize(
    "num_samples, fun, atol, max_iterations",
    [
        (5, group_testing_shapley, 0.01, int(1e5)),
    ],
)
def test_group_testing_shapley(
    num_samples, analytic_shapley, fun, atol, max_iterations
):
    u, exact_values = analytic_shapley
    values = fun(u, max_iterations=int(max_iterations), progress=False, n_jobs=1)
    check_values(values, exact_values, atol=atol)

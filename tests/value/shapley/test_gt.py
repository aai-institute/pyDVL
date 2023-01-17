import pytest

from pydvl.utils.status import Status
from pydvl.value.shapley.gt import group_testing_shapley

from .. import check_values


@pytest.mark.parametrize(
    "num_samples, fun, atol, n_iterations, kwargs",
    [(5, group_testing_shapley, 0.05, int(1e5), {"eps": 0.05})],
)
def test_group_testing_shapley(
    num_samples, analytic_shapley, fun, atol, n_iterations, kwargs
):
    u, exact_values = analytic_shapley
    values = fun(u, n_iterations=int(n_iterations), progress=False, n_jobs=1, **kwargs)
    assert values.status == Status.Converged
    check_values(values, exact_values, atol=atol)

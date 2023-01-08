import pytest

from pydvl.value import ValuationStatus
from pydvl.value.shapley.gt import group_testing_shapley

from .. import check_values


@pytest.mark.parametrize(
    "num_samples, fun, atol, max_iterations, kwargs",
    [(5, group_testing_shapley, 0.05, int(1e5), {"eps": 0.05})],
)
def test_group_testing_shapley(
    num_samples, analytic_shapley, fun, atol, max_iterations, kwargs
):
    u, exact_values = analytic_shapley
    values = fun(
        u, max_iterations=int(max_iterations), progress=False, n_jobs=1, **kwargs
    )
    assert values.status == ValuationStatus.Converged
    check_values(values, exact_values, atol=atol)

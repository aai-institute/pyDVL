import pytest

from pydvl.value.loo import naive_loo

from .. import check_total_value, check_values


@pytest.mark.parametrize("num_samples", [0, 10])
def test_naive_loo(num_samples, analytic_loo):
    """Compares the naive loo with analytic values in a dummy model"""
    u, exact_values = analytic_loo
    values = naive_loo(u, progress=False)
    check_total_value(u, values, rtol=0.1)
    check_values(values, exact_values, rtol=0.1)

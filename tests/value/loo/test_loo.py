import pytest

from pydvl.value.loo import naive_loo

from ..conftest import check_total_value, check_values


@pytest.mark.xfail(reason="Test ")
def test_naive_loo(analytic_loo):
    """Compares the naive loo with analytic values in a dummy model"""
    u, exact_values = analytic_loo
    values_p = naive_loo(u, progress=False)
    check_total_value(u, values_p, rtol=0.1)
    check_values(values_p, exact_values, rtol=0.1)

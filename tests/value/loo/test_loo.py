import pytest

from pydvl.value.loo import loo

from .. import check_total_value, check_values


@pytest.mark.parametrize("num_samples", [10, 100])
def test_loo(num_samples: int, n_jobs: int, parallel_config, analytic_loo):
    """Compares the naive loo with analytic values in a dummy model"""
    u, exact_values = analytic_loo
    values = loo(u, n_jobs=n_jobs, config=parallel_config, progress=False)
    check_total_value(u, values, rtol=0.1)
    check_values(values, exact_values, rtol=0.1)

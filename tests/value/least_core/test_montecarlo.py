import logging

import pytest
from numpy.testing import assert_almost_equal

from pydvl.value.least_core import montecarlo_least_core
from tests.value import check_values

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "test_utility, rtol, n_iterations",
    [
        (("miner", {"n_miners": 8}), 0.1, 128),
        (("gloves", {"left": 10, "right": 5}), 0.2, 10000),
    ],
    indirect=["test_utility"],
)
@pytest.mark.parametrize("n_jobs", [1, -1])
@pytest.mark.parametrize("non_negative_subsidy", (True, False))
def test_montecarlo_least_core(
    test_utility, rtol, n_iterations, n_jobs, non_negative_subsidy, seed
):
    u, exact_values = test_utility

    values = montecarlo_least_core(
        u,
        n_iterations=n_iterations,
        non_negative_subsidy=non_negative_subsidy,
        progress=False,
        n_jobs=n_jobs,
        seed=seed,
    )
    if non_negative_subsidy:
        check_values(values, exact_values)
        # Sometimes the subsidy is negative but really close to zero
        # Due to numerical errors
        if values.subsidy < 0:
            assert_almost_equal(values.subsidy, 0.0, decimal=5)
    else:
        check_values(values, exact_values, extra_values_names=["subsidy"])

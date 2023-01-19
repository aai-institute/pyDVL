import logging

import pytest

from pydvl.value.least_core import montecarlo_least_core
from tests.value import check_values

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "test_utility, rtol, n_iterations",
    [
        (("miner", {"n_miners": 8}), 0.1, 128),
        (("miner", {"n_miners": 15}), 0.15, 6000),
        (("gloves", {"left": 4, "right": 4}), 0.1, 128),
        (("gloves", {"left": 10, "right": 5}), 0.15, 6000),
    ],
    indirect=["test_utility"],
)
@pytest.mark.parametrize(
    "n_jobs",
    [
        1,
        4,
        pytest.param(
            -1, marks=pytest.mark.skip("Skipping n_jobs=-1 until it is fixed")
        ),
    ],
)
def test_montecarlo_least_core(test_utility, rtol, n_iterations, n_jobs):
    u, exact_values = test_utility

    values = montecarlo_least_core(
        u, n_iterations=n_iterations, progress=False, n_jobs=n_jobs
    )
    check_values(values, exact_values, rtol=rtol, extra_values_names=["subsidy"])


@pytest.mark.parametrize(
    "test_utility, n_iterations",
    [
        (("miner", {"n_miners": 8}), 3),
    ],
    indirect=["test_utility"],
)
def test_montecarlo_least_core_failure(test_utility, n_iterations):
    u, exact_values = test_utility

    with pytest.raises(ValueError):
        montecarlo_least_core(u, n_iterations=n_iterations, progress=False)

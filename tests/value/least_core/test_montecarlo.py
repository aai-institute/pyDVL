import logging

import pytest

from pydvl.value.least_core import montecarlo_least_core
from tests.value import check_values

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "miner_utility, rtol, max_iterations",
    [
        (8, 0.1, 128),
        (15, 0.15, 4096),
    ],
    indirect=["miner_utility"],
)
def test_montecarlo_least_core(miner_utility, rtol, max_iterations):
    u, exact_values = miner_utility

    values = montecarlo_least_core(
        u, max_iterations=max_iterations, progress=False, n_jobs=4
    )
    check_values(values, exact_values, rtol=rtol)


@pytest.mark.parametrize(
    "miner_utility, max_iterations",
    [
        (8, 3),
    ],
    indirect=["miner_utility"],
)
def test_montecarlo_least_core_failure(miner_utility, max_iterations):
    u, exact_values = miner_utility

    with pytest.raises(ValueError):
        montecarlo_least_core(
            u, max_iterations=max_iterations, progress=False, n_jobs=4
        )

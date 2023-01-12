import logging

import numpy as np
import pytest

from pydvl.value.least_core import montecarlo_least_core
from tests.value import check_values

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "miner_utility, rtol, n_iterations",
    [
        (8, 0.1, 128),
        (15, 0.15, 6000),
    ],
    indirect=["miner_utility"],
)
def test_montecarlo_least_core(miner_utility, rtol, n_iterations):
    u, exact_values = miner_utility

    values = montecarlo_least_core(
        u, n_iterations=n_iterations, progress=False, n_jobs=4
    )
    check_values(
        values, exact_values, rtol=rtol, extra_values_names=["least_core_value"]
    )


@pytest.mark.parametrize(
    "miner_utility, n_iterations",
    [
        (8, 3),
    ],
    indirect=["miner_utility"],
)
def test_montecarlo_least_core_failure(miner_utility, n_iterations):
    u, exact_values = miner_utility

    with pytest.raises(ValueError):
        montecarlo_least_core(u, n_iterations=n_iterations, progress=False, n_jobs=4)

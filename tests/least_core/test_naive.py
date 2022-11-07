import logging

import pytest

from pydvl.least_core import exact_least_core
from tests.conftest import check_total_value, check_values

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "miner_utility",
    [3, 4, 8, 9],
    indirect=True,
)
def test_naive_least_core(miner_utility):
    u, exact_values = miner_utility
    values = exact_least_core(u, progress=False)
    check_total_value(u, values)
    check_values(values, exact_values)

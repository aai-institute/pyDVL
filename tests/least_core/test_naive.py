import logging

import pytest

from pydvl.least_core import naive_lc
from tests.conftest import check_total_value, check_values

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "miner_utility",
    [3, 4, 8, 9],
    indirect=True,
)
def test_naive_least_core(miner_utility):
    u, exact_values = miner_utility
    values = naive_lc(u, progress=False)
    check_total_value(u, values)
    check_values(values, exact_values)

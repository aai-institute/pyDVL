import pytest

from pydvl.value.least_core import exact_least_core
from tests.value import check_total_value, check_values


@pytest.mark.parametrize(
    "test_utility",
    [
        ("miner", {"n_miners": 3}),
        ("miner", {"n_miners": 4}),
        ("miner", {"n_miners": 8}),
        ("miner", {"n_miners": 9}),
        ("gloves", {"left": 1, "right": 1}),
        ("gloves", {"left": 2, "right": 1}),
        ("gloves", {"left": 1, "right": 2}),
        ("gloves", {"left": 3, "right": 1}),
    ],
    indirect=True,
)
def test_naive_least_core(test_utility):
    u, exact_values = test_utility
    values = exact_least_core(u, progress=False)
    check_total_value(u, values)
    check_values(values, exact_values, extra_values_names=["subsidy"])

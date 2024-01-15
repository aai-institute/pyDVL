import pytest
from numpy.testing import assert_almost_equal

from pydvl.value.least_core import exact_least_core
from tests.value import check_total_value, check_values


@pytest.mark.parametrize(
    "test_game",
    [
        ("miner", {"n_players": 3}),
        ("miner", {"n_players": 4}),
        ("gloves", {"left": 1, "right": 1}),
        ("gloves", {"left": 2, "right": 1}),
        ("gloves", {"left": 1, "right": 2}),
    ],
    indirect=True,
)
@pytest.mark.parametrize("non_negative_subsidy", (True, False))
def test_naive_least_core(test_game, non_negative_subsidy):
    values = exact_least_core(
        test_game.u, non_negative_subsidy=non_negative_subsidy, progress=False
    )
    check_total_value(test_game.u, values)
    exact_values = test_game.least_core_values()
    if non_negative_subsidy:
        check_values(values, exact_values)
        # Sometimes the subsidy is negative but really close to zero
        # Due to numerical errors
        if values.subsidy < 0:
            assert_almost_equal(values.subsidy, 0.0, decimal=5)
    else:
        check_values(values, exact_values, extra_values_names=["subsidy"])

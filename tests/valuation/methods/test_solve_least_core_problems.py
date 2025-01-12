import pytest

from pydvl.utils import Status
from pydvl.valuation.methods._solve_least_core_problems import lc_solve_problem

from .. import check_values


@pytest.mark.parametrize(
    "test_game",
    [("miner", {"n_players": 4})],
    indirect=True,
)
def test_lc_solve_problems(test_game):
    test_game.u = test_game.u.with_dataset(test_game.data)
    problem = test_game.least_core_problem()

    exact_values = test_game.least_core_values()

    solution = lc_solve_problem(problem, u=test_game.u, algorithm="test_lc")
    assert solution.status == Status.Converged
    check_values(solution, exact_values, rtol=0.01)

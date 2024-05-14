import pytest

from pydvl.utils import Status
from pydvl.valuation.methods._least_core_solving import (
    lc_solve_problem,
    lc_solve_problems,
)

from .. import check_values


@pytest.mark.parametrize(
    "test_game",
    [("miner", {"n_players": 4})],
    indirect=True,
)
def test_lc_solve_problems(test_game, n_jobs, parallel_backend):
    """Test solving LeastCoreProblems in parallel."""

    test_game.u = test_game.u.with_dataset(test_game.data)
    n_problems = n_jobs
    problem = test_game.least_core_problem()
    solutions = lc_solve_problems(
        [problem] * n_problems,
        test_game.u,
        algorithm="test_lc",
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
    )
    assert len(solutions) == n_problems

    exact_values = test_game.least_core_values()

    for solution in solutions:
        assert solution.status == Status.Converged
        check_values(solution, exact_values, rtol=0.01)

        check = lc_solve_problem(problem, u=test_game.u, algorithm="test_lc")
        assert check.status == Status.Converged
        check_values(solution, check, rtol=0.01)

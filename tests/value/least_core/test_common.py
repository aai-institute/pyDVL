import pytest

from pydvl.utils import Status
from pydvl.value.least_core.common import lc_solve_problem, lc_solve_problems
from pydvl.value.least_core.naive import lc_prepare_problem

from .. import check_values


@pytest.mark.parametrize(
    "test_game",
    [("miner", {"n_players": 5})],
    indirect=True,
)
def test_lc_solve_problems(test_game, n_jobs, parallel_backend):
    """Test solving LeastCoreProblems in parallel."""

    n_problems = n_jobs
    problem = lc_prepare_problem(test_game.u)
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

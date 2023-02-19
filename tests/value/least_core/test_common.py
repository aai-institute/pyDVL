import pytest

from pydvl.utils import Status
from pydvl.value.least_core.common import lc_solve_problem, lc_solve_problems
from pydvl.value.least_core.naive import lc_prepare_problem

from .. import check_values


@pytest.mark.parametrize(
    "test_utility",
    [("miner", {"n_miners": 5})],
    indirect=True,
)
def test_lc_solve_problems(test_utility, n_jobs, parallel_config):
    """Test solving LeastCoreProblems in parallel."""

    u, exact_values = test_utility
    n_problems = n_jobs
    problem = lc_prepare_problem(u)
    solutions = lc_solve_problems(
        [problem] * n_problems,
        u,
        algorithm="test_lc",
        n_jobs=n_jobs,
        config=parallel_config,
    )
    assert len(solutions) == n_problems

    for solution in solutions:
        assert solution.status == Status.Converged
        check_values(solution, exact_values, rtol=0.01)

        check = lc_solve_problem(problem, u=u, algorithm="test_lc")
        assert check.status == Status.Converged
        check_values(solution, check, rtol=0.01)

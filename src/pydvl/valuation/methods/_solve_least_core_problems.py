from __future__ import annotations

import logging
import warnings
from typing import NamedTuple, Tuple, cast

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from pydvl.utils import Status
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.types import Sample
from pydvl.valuation.utility.base import UtilityBase

__all__ = [
    "_solve_least_core_linear_program",
    "_solve_egalitarian_least_core_quadratic_program",
    "lc_solve_problem",
    "LeastCoreProblem",
]

logger = logging.getLogger(__name__)


class LeastCoreProblem(NamedTuple):
    """The linear programming problem that defines the least-core valuation.

    See [LeastCoreValuation][pydvl.valuation.methods.least_core.LeastCoreValuation] for
    details.

    """

    utility_values: NDArray[np.float64]
    A_lb: NDArray[np.float64]


def lc_solve_problem(
    problem: LeastCoreProblem,
    *,
    u: UtilityBase,
    algorithm: str,
    non_negative_subsidy: bool = False,
    solver_options: dict | None = None,
) -> ValuationResult:
    """Solves a linear problem as prepared by
    [mclc_prepare_problem()][pydvl.valuation.create_least_core_problem].
    Useful for parallel execution of multiple experiments by running this as a
    remote task.

    Args:
        problem: LeastCoreProblem to solve.
        u: Utility object with model, data and scoring function.
        algorithm: Name of the least-core valuation algorithm.
        non_negative_subsidy: If True, the least core subsidy $e$ is constrained
            to be non-negative.
        solver_options: Optional dictionary containing a CVXPY solver and options to
            configure it. For valid values to the "solver" key see
            [here](https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver).
            For additional options see [here](https://www.cvxpy.org/tutorial/solvers/index.html#setting-solver-options).

    """
    if u.training_data is None:
        raise ValueError("Utility object must have a training dataset.")

    n_obs = len(u.training_data)

    if np.any(np.isnan(problem.utility_values)):
        warnings.warn(
            f"Calculation returned "
            f"{np.sum(np.isnan(problem.utility_values))} NaN "
            f"values out of {problem.utility_values.size}",
            RuntimeWarning,
        )

    if solver_options is None:
        solver_options = {}
    else:
        solver_options = solver_options.copy()

    if "solver" not in solver_options:
        solver_options["solver"] = cp.SCS

    if "max_iters" not in solver_options and solver_options["solver"] == cp.SCS:
        solver_options["max_iters"] = 10000

    logger.debug("Removing possible duplicate values in lower bound array")
    b_lb = problem.utility_values
    A_lb, unique_indices = np.unique(problem.A_lb, return_index=True, axis=0)
    b_lb = b_lb[unique_indices]

    logger.debug("Building equality constraint")
    A_eq = np.ones((1, n_obs))
    # We might have already computed the total utility one or more times.
    # This is the index of the row(s) in A_lb with all ones.
    total_utility_indices = np.where(A_lb.sum(axis=1) == n_obs)[0]
    if len(total_utility_indices) == 0:
        b_eq = np.array([u(Sample(idx=None, subset=u.training_data.indices))])
    else:
        b_eq = b_lb[total_utility_indices]
        # Remove the row(s) corresponding to the total utility
        # from the lower bound constraints
        # because given the equality constraint
        # it is the same as using the constraint e >= 0
        # (i.e. setting non_negative_subsidy = True).
        mask: NDArray[np.bool_] = np.ones_like(b_lb, dtype=bool)
        mask[total_utility_indices] = False
        b_lb = b_lb[mask]
        A_lb = A_lb[mask]

    # Remove the row(s) corresponding to the empty subset
    # because, given u(∅) = (which is almost always the case,
    # it is the same as using the constraint e >= 0
    # (i.e. setting non_negative_subsidy = True).
    emptyset_utility_indices = np.where(A_lb.sum(axis=1) == 0)[0]
    if len(emptyset_utility_indices) > 0:
        mask = np.ones_like(b_lb, dtype=bool)
        mask[emptyset_utility_indices] = False
        b_lb = b_lb[mask]
        A_lb = A_lb[mask]

    _, subsidy = _solve_least_core_linear_program(
        A_eq=A_eq,
        b_eq=b_eq,
        A_lb=A_lb,
        b_lb=b_lb,
        non_negative_subsidy=non_negative_subsidy,
        solver_options=solver_options,
    )

    values: NDArray[np.float64] | None

    if subsidy is None:
        logger.debug("No values were found")
        status = Status.Failed
        values = np.empty(n_obs)
        values[:] = np.nan
        subsidy = np.nan
    else:
        values = _solve_egalitarian_least_core_quadratic_program(
            subsidy,
            A_eq=A_eq,
            b_eq=b_eq,
            A_lb=A_lb,
            b_lb=b_lb,
            solver_options=solver_options,
        )

        if values is None:
            logger.debug("No values were found")
            status = Status.Failed
            values = np.empty(n_obs)
            values[:] = np.nan
            subsidy = np.nan
        else:
            status = Status.Converged

    return ValuationResult(
        algorithm=algorithm,
        status=status,
        values=values,
        subsidy=subsidy,
        stderr=None,
        data_names=u.training_data.names,
    )


def _solve_least_core_linear_program(
    A_eq: NDArray[np.float64],
    b_eq: NDArray[np.float64],
    A_lb: NDArray[np.float64],
    b_lb: NDArray[np.float64],
    solver_options: dict,
    non_negative_subsidy: bool = False,
) -> Tuple[NDArray[np.float64] | None, float | None]:
    r"""Solves the Least Core's linear program using cvxopt.

    $$
        \text{minimize} \ & e \\
        \mbox{such that} \ & A_{eq} x = b_{eq}, \\
        & A_{lb} x + e \ge b_{lb},\\
        & A_{eq} x = b_{eq},\\
        & x in \mathcal{R}^n , \\
    $$
     where $x$ is a vector of decision variables; ,
    $b_{ub}$, $b_{eq}$, $l$, and $u$ are vectors; and
    $A_{ub}$ and $A_{eq}$ are matrices.

    if `non_negative_subsidy` is True, then an additional constraint $e \ge 0$ is used.

    Args:
        A_eq: The equality constraint matrix. Each row of `A_eq` specifies the
            coefficients of a linear equality constraint on `x`.
        b_eq: The equality constraint vector. Each element of `A_eq @ x` must equal
            the corresponding element of `b_eq`.
        A_lb: The inequality constraint matrix. Each row of `A_lb` specifies the
            coefficients of a linear inequality constraint on `x`.
        b_lb: The inequality constraint vector. Each element represents a
            lower bound on the corresponding value of `A_lb @ x`.
            non_negative_subsidy: If True, the least core subsidy $e$ is constrained
            to be non-negative.
        options: Keyword arguments that will be used to select a solver
            and to configure it. For all possible options, refer to [cvxpy's
            documentation](https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options).
    """
    logger.debug(f"Solving linear program : {A_eq=}, {b_eq=}, {A_lb=}, {b_lb=}")

    n_variables = A_eq.shape[1]

    x = cp.Variable(n_variables)
    e = cp.Variable()

    objective = cp.Minimize(e)
    constraints = [A_eq @ x == b_eq, (A_lb @ x + e * np.ones(len(A_lb))) >= b_lb]

    if non_negative_subsidy:
        constraints += [e >= 0]

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(**solver_options)
    except cp.error.SolverError as err:
        raise ValueError("Could not solve linear program") from err

    if problem.status in cp.settings.SOLUTION_PRESENT:
        logger.debug("Problem was solved")
        if problem.status == cp.settings.USER_LIMIT:
            warnings.warn(
                "Solver terminated early. Consider increasing the solver's "
                "maximum number of iterations in solver_options",
                RuntimeWarning,
            )
        subsidy = cast(NDArray[np.float64], e.value).item()
        return x.value, subsidy

    if problem.status in cp.settings.INF_OR_UNB:
        warnings.warn(
            "Could not find solution due to infeasibility or unboundedness of problem.",
            RuntimeWarning,
        )
    return None, None


def _solve_egalitarian_least_core_quadratic_program(
    subsidy: float,
    A_eq: NDArray[np.float64],
    b_eq: NDArray[np.float64],
    A_lb: NDArray[np.float64],
    b_lb: NDArray[np.float64],
    solver_options: dict,
) -> NDArray[np.float64] | None:
    r"""Solves the egalitarian Least Core's quadratic program using cvxopt.

    $$
        \text{minimize} \ & \| x \|_2 \\
        \mbox{such that} \ & A_{eq} x = b_{eq}, \\
        & A_{lb} x + e \ge b_{lb},\\
        & A_{eq} x = b_{eq},\\
        & x in \mathcal{R}^n , \\
        & e \text{ is a constant.}
    $$
     where $x$ is a vector of decision variables; ,
    $b_{ub}$, $b_{eq}$, $l$, and $u$ are vectors; and
    $A_{ub}$ and $A_{eq}$ are matrices.

    Args:
        subsidy: Minimal subsidy returned by
            [_solve_least_core_linear_program()][pydvl.valuation.methods._solve_least_core_problems._solve_least_core_linear_program].
        A_eq: The equality constraint matrix. Each row of `A_eq` specifies the
            coefficients of a linear equality constraint on `x`.
        b_eq: The equality constraint vector. Each element of `A_eq @ x` must equal
            the corresponding element of `b_eq`.
        A_lb: The inequality constraint matrix. Each row of `A_lb` specifies the
            coefficients of a linear inequality constraint on `x`.
        b_lb: The inequality constraint vector. Each element represents a
            lower bound on the corresponding value of `A_lb @ x`.
        solver_options: Keyword arguments that will be used to select a solver
            and to configure it. Refer to [cvxpy's
            documentation](https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options)
            for all possible options.
    """
    logger.debug(f"Solving quadratic program : {A_eq=}, {b_eq=}, {A_lb=}, {b_lb=}")

    n_variables = A_eq.shape[1]

    x = cp.Variable(n_variables)

    objective = cp.Minimize(cp.norm2(x))
    constraints = [A_eq @ x == b_eq, (A_lb @ x + subsidy * np.ones(len(A_lb))) >= b_lb]
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(**solver_options)
    except cp.error.SolverError as err:
        raise ValueError("Could not solve quadratic program") from err

    if problem.status in cp.settings.SOLUTION_PRESENT:
        logger.debug("Problem was solved")
        if problem.status == cp.settings.USER_LIMIT:
            warnings.warn(
                "Solver terminated early. Consider increasing the solver's "
                "maximum number of iterations in solver_options",
                RuntimeWarning,
            )
        return x.value  # type: ignore

    if problem.status in cp.settings.INF_OR_UNB:
        warnings.warn(
            "Could not find solution due to infeasibility or unboundedness of problem.",
            RuntimeWarning,
        )
    return None

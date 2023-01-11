import logging
import warnings
from typing import Optional, Tuple

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

__all__ = ["_solve_least_core_linear_program"]

logger = logging.getLogger(__name__)


def _solve_least_core_linear_program(
    n_variables: int,
    A_eq: NDArray[np.float_],
    b_eq: NDArray[np.float_],
    A_lb: NDArray[np.float_],
    b_lb: NDArray[np.float_],
    **options,
) -> Tuple[Optional[NDArray[np.float_]], Optional[float]]:
    """Solves the Least Core's linear program using cvxopt.

    .. math::

        \text{minimize} \ & e \\
        \mbox{such that} \ & A_{eq} x = b_{eq}, \\
        & A_{lb} x + e \ge b_{lb},\\
        & A_{eq} x = b_{eq},\\
        & x in \mathcal{R}^n , \\
        & e \ge 0

     where :math:`x` is a vector of decision variables; ,
    :math:`b_{ub}`, :math:`b_{eq}`, :math:`l`, and :math:`u` are vectors; and
    :math:`A_{ub}` and :math:`A_{eq}` are matrices.

    :param A_eq: The equality constraint matrix. Each row of ``A_eq`` specifies the
        coefficients of a linear equality constraint on ``x``.
    :param b_eq: The equality constraint vector. Each element of ``A_eq @ x`` must equal
        the corresponding element of ``b_eq``.
    :param A_ub: The inequality constraint matrix. Each row of ``A_ub`` specifies the
        coefficients of a linear inequality constraint on ``x``.
    :param b_ub: The inequality constraint vector. Each element represents an
        upper bound on the corresponding value of ``A_ub @ x``.
    :param options: Keyword arguments that will be used to select a solver
        and to configure it. Refer to the following page for all possible options:
        https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
    """
    logger.debug(f"Solving linear program : {A_eq=}, {b_eq=}, {A_lb=}, {b_lb=}")

    x = cp.Variable(n_variables)
    e = cp.Variable()
    objective = cp.Minimize(e)
    constraints = [
        e >= 0,
        A_eq @ x == b_eq,
        A_lb @ x + e * np.ones(len(A_lb)) >= b_lb,
    ]
    problem = cp.Problem(objective, constraints)

    solver = options.pop("solver", cp.ECOS)

    try:
        problem.solve(solver=solver, **options)
    except cp.SOLVER_ERROR as e:
        raise ValueError("Could not solve linear program") from e

    if problem.status in cp.settings.SOLUTION_PRESENT:
        logger.debug("Problem was solved")
        if problem.status in [cp.settings.OPTIMAL_INACCURATE, cp.settings.USER_LIMIT]:
            warnings.warn(
                "Solver terminated early. Consider increasing the solver's "
                "maximum number of iterations in options",
                RuntimeWarning,
            )
        return x.value, e.value.item()

    if problem.status in cp.settings.INF_OR_UNB:
        warnings.warn(
            "Could not find solution due to infeasibility or unboundedness of problem.",
            RuntimeWarning,
        )
    return None, None

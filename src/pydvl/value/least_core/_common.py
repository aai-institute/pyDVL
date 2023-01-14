import logging
import warnings
from typing import Optional, Tuple

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "_solve_least_core_linear_program",
    "_solve_egalitarian_least_core_quadratic_program",
]

logger = logging.getLogger(__name__)


def _solve_least_core_linear_program(
    A_eq: NDArray[np.float_],
    b_eq: NDArray[np.float_],
    A_lb: NDArray[np.float_],
    b_lb: NDArray[np.float_],
    *,
    epsilon: float = 0.0,
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
    :param A_lb: The inequality constraint matrix. Each row of ``A_lb`` specifies the
        coefficients of a linear inequality constraint on ``x``.
    :param b_lb: The inequality constraint vector. Each element represents a
        lower bound on the corresponding value of ``A_lb @ x``.
    :param epsilon: Relaxation value by which the subset utility is decreased.
    :param options: Keyword arguments that will be used to select a solver
        and to configure it. For all possible options, refer to `cvxpy's documentation
        <https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options>`_
    """
    logger.debug(f"Solving linear program : {A_eq=}, {b_eq=}, {A_lb=}, {b_lb=}")

    n_variables = A_eq.shape[1]

    x = cp.Variable(n_variables)
    e = cp.Variable()
    epsilon_parameter = cp.Parameter(name="epsilon", nonneg=True, value=epsilon)

    objective = cp.Minimize(e)
    constraints = [
        e >= 0,
        A_eq @ x == b_eq,
        (A_lb @ x + e * np.ones(len(A_lb))) >= (b_lb - epsilon_parameter),
    ]
    problem = cp.Problem(objective, constraints)

    solver = options.pop("solver", cp.ECOS)

    try:
        problem.solve(solver=solver, **options)
    except cp.error.SolverError as err:
        raise ValueError("Could not solve linear program") from err

    if problem.status in cp.settings.SOLUTION_PRESENT:
        logger.debug("Problem was solved")
        if problem.status in [cp.settings.OPTIMAL_INACCURATE, cp.settings.USER_LIMIT]:
            warnings.warn(
                "Solver terminated early. Consider increasing the solver's "
                "maximum number of iterations in options",
                RuntimeWarning,
            )
        subsidy = e.value.item()
        # HACK: sometimes the returned least core subsidy
        # is negative but very close to 0
        # to avoid any problems with the subsequent quadratic program
        # we just set it to 0.0
        if subsidy < 0:
            warnings.warn(
                f"Least core subsidy e={subsidy} is negative but close to zero. "
                "It will be set to 0.0",
                RuntimeWarning,
            )
            subsidy = 0.0
        return x.value, subsidy

    if problem.status in cp.settings.INF_OR_UNB:
        warnings.warn(
            "Could not find solution due to infeasibility or unboundedness of problem.",
            RuntimeWarning,
        )
    return None, None


def _solve_egalitarian_least_core_quadratic_program(
    subsidy: float,
    A_eq: NDArray[np.float_],
    b_eq: NDArray[np.float_],
    A_lb: NDArray[np.float_],
    b_lb: NDArray[np.float_],
    epsilon: float = 0.0,
    **options,
) -> Optional[NDArray[np.float_]]:
    """Solves the egalitarian Least Core's quadratic program using cvxopt.

    .. math::

        \text{minimize} \ & \| x \|_2 \\
        \mbox{such that} \ & A_{eq} x = b_{eq}, \\
        & A_{lb} x + e \ge b_{lb},\\
        & A_{eq} x = b_{eq},\\
        & x in \mathcal{R}^n , \\
        & e \text{ is a constant.}

     where :math:`x` is a vector of decision variables; ,
    :math:`b_{ub}`, :math:`b_{eq}`, :math:`l`, and :math:`u` are vectors; and
    :math:`A_{ub}` and :math:`A_{eq}` are matrices.

    :param subsidy: Minimal subsidy returned by :func:`_solve_least_core_linear_program`
    :param A_eq: The equality constraint matrix. Each row of ``A_eq`` specifies the
        coefficients of a linear equality constraint on ``x``.
    :param b_eq: The equality constraint vector. Each element of ``A_eq @ x`` must equal
        the corresponding element of ``b_eq``.
    :param A_lb: The inequality constraint matrix. Each row of ``A_lb`` specifies the
        coefficients of a linear inequality constraint on ``x``.
    :param b_lb: The inequality constraint vector. Each element represents a
        lower bound on the corresponding value of ``A_lb @ x``.
    :param epsilon: Relaxation value by which the subset utility is decreased.
    :param options: Keyword arguments that will be used to select a solver
        and to configure it. Refer to the following page for all possible options:
        https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
    """
    logger.debug(f"Solving quadratic program : {A_eq=}, {b_eq=}, {A_lb=}, {b_lb=}")

    if subsidy < 0:
        raise ValueError("The least core subsidy must be non-negative.")

    n_variables = A_eq.shape[1]

    x = cp.Variable(n_variables)
    epsilon_parameter = cp.Parameter(name="epsilon", nonneg=True, value=epsilon)

    objective = cp.Minimize(cp.norm2(x))
    constraints = [
        A_eq @ x == b_eq,
        (A_lb @ x + subsidy * np.ones(len(A_lb))) >= (b_lb - epsilon_parameter),
    ]
    problem = cp.Problem(objective, constraints)

    solver = options.pop("solver", cp.ECOS)

    try:
        problem.solve(solver=solver, **options)
    except cp.error.SolverError as err:
        raise ValueError("Could not solve quadratic program") from err

    if problem.status in cp.settings.SOLUTION_PRESENT:
        logger.debug("Problem was solved")
        if problem.status in [cp.settings.OPTIMAL_INACCURATE, cp.settings.USER_LIMIT]:
            warnings.warn(
                "Solver terminated early. Consider increasing the solver's "
                "maximum number of iterations in options",
                RuntimeWarning,
            )
        return x.value  # type: ignore

    if problem.status in cp.settings.INF_OR_UNB:
        warnings.warn(
            "Could not find solution due to infeasibility or unboundedness of problem.",
            RuntimeWarning,
        )
    return None

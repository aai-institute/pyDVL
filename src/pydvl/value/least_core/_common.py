import logging
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy
from numpy.typing import NDArray

__all__ = ["_solve_linear_program"]

logger = logging.getLogger(__name__)


BOUNDS_TYPE = Union[
    Tuple[Optional[float], Optional[float]],
    List[Tuple[Optional[float], Optional[float]]],
]


def _solve_linear_program(
    c: NDArray[np.float_],
    A_eq: NDArray[np.float_],
    b_eq: NDArray[np.float_],
    A_ub: NDArray[np.float_],
    b_ub: NDArray[np.float_],
    bounds: BOUNDS_TYPE,
    **options,
) -> Optional[NDArray[np.float_]]:
    """Solves a linear program using scipy's :func:`~scipy.optimize.linprog`
    function.

    .. note::
       The following description of the linear program and the parameters is
       taken verbatim from scipy

    .. math::

        \min_x \ & c^T x \\
        \mbox{such that} \ & A_{ub} x \leq b_{ub},\\
        & A_{eq} x = b_{eq},\\
        & l \leq x \leq u ,

    where $x$ is a vector of decision variables; $c$, $b_{ub}$, $b_{eq}$, $l$,
    and $u$ are vectors, and $A_{ub}$ and $A_{eq}$ are matrices.

    :param c: The coefficients of the linear objective function to be minimized.
    :param A_eq: The equality constraint matrix. Each row of ``A_eq`` specifies
        the coefficients of a linear equality constraint on ``x``.
    :param b_eq: The equality constraint vector. Each element of ``A_eq @ x``
        must equal the corresponding element of ``b_eq``.
    :param A_ub: The inequality constraint matrix. Each row of ``A_ub``
        specifies the coefficients of a linear inequality constraint on ``x``.
    :param b_ub: The inequality constraint vector. Each element represents an
        upper bound on the corresponding value of ``A_ub @ x``.
    :param bounds: A sequence of ``(min, max)`` pairs for each element in ``x``,
        defining the minimum and maximum values of that decision variable. Use
        ``None`` to indicate that there is no bound. By default, bounds are
        ``(0, None)`` (all decision variables are non-negative). If a single
        tuple ``(min, max)`` is provided, then ``min`` and ``max`` will serve as
        bounds for all decision variables.
    :param options: A dictionary of solver options. Refer to scipy's
        documentation for all possible values.
    """
    logger.debug(
        f"Solving linear programming problem: {c=}, {A_eq=}, {b_eq=}, {A_ub=}, {b_ub=}"
    )

    result: scipy.optimize.OptimizeResult = scipy.optimize.linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs-ipm",
        options=options,
    )

    logger.debug(f"{result=}")

    if result.success:
        return np.asarray(result.x)

    values = None

    if result.status == 1:
        warnings.warn(
            f"Solver terminated early: '{result.message}'. Consider increasing the solver's maxiter in options"
        )
    elif result.status == 2:
        warnings.warn(
            f"Could not find solution due to infeasibility of problem: '{result.message}'. "
            "Consider increasing max_iterations",
            RuntimeWarning,
        )
    elif result.status == 3:
        warnings.warn(
            f"Could not find solution due to unboundedness of problem: '{result.message}'. "
            "Consider increasing max_iterations",
            RuntimeWarning,
        )
    else:
        warnings.warn(
            f"Could not find solution due to numerical issues: '{result.message}'. "
            "Consider increasing max_iterations",
            RuntimeWarning,
        )
    return values

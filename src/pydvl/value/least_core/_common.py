import logging
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy
from numpy.typing import NDArray

__all__ = ["_solve_linear_programming"]

logger = logging.getLogger(__name__)


BOUNDS_TYPE = Union[
    Tuple[Optional[float], Optional[float]],
    List[Tuple[Optional[float], Optional[float]]],
]


def _solve_linear_programming(
    c: NDArray[np.float_],
    A_eq: NDArray[np.float_],
    b_eq: NDArray[np.float_],
    A_ub: NDArray[np.float_],
    b_ub: NDArray[np.float_],
    bounds: BOUNDS_TYPE,
    **options,
) -> Optional[NDArray[np.float_]]:
    """Solves a linear programming problem using cvxopt.

    :param c:
    :param A_eq:
    :param b_eq:
    :param A_ub:
    :param b_ub:
    :param bounds:
    :param options:
    """
    logger.debug("Solving linear programming problem")
    logger.debug(f"{c=}")
    logger.debug(f"{A_eq=}")
    logger.debug(f"{b_eq=}")
    logger.debug(f"{A_ub=}")
    logger.debug(f"{b_ub=}")

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
        # We select all but the last entry because it represents the least core value 'e'
        values = np.asarray(result.x[:-1])
    else:
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
        values = None
    return values

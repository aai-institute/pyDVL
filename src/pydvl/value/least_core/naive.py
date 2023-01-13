import logging
import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from pydvl.utils import Utility, maybe_progress, powerset
from pydvl.value.least_core._common import (
    _solve_egalitarian_least_core_quadratic_program,
    _solve_least_core_linear_program,
)
from pydvl.value.results import ValuationResult, ValuationStatus

__all__ = ["exact_least_core"]

logger = logging.getLogger(__name__)


def exact_least_core(
    u: Utility, *, options: Optional[dict] = None, progress: bool = True, **kwargs
) -> ValuationResult:
    r"""Computes the exact Least Core values.

    .. note::
       If the training set contains more than 20 instances a warning is printed
       because the computation is very expensive. This method is mostly used for
       internal testing and simple use cases. Please refer to the
       :func:`Monte Carlo method <pydvl.value.least_core.montecarlo.montecarlo_least_core>`
       for practical applications.

    The least core is the solution to the following Linear Programming problem:

    $$
    \begin{array}{lll}
    \text{minimize} & \displaystyle{e} & \\
    \text{subject to} & \displaystyle\sum_{i\in N} x_{i} = v(N) & \\
    & \displaystyle\sum_{i\in S} x_{i} + e \geq v(S) &, \forall S \subseteq N \\
    \end{array}
    $$

    Where $N = \{1, 2, \dots, n\}$ are the training set's indices.

    :param u: Utility object with model, data, and scoring function
    :param options: Keyword arguments that will be used to select a solver
        and to configure it. Refer to the following page for all possible options:
        https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
    :param progress: If True, shows a tqdm progress bar

    :return: Object with the data values and the least core value.
    """
    n = len(u.data)

    # Arbitrary choice, will depend on time required, caching, etc.
    if n > 20:
        warnings.warn(f"Large dataset! Computation requires 2^{n} calls to model.fit()")

    if options is None:
        options = {}

    powerset_size = 2**n

    logger.debug("Building vectors and matrices for linear programming problem")
    A_eq = np.ones((1, n))
    A_lb = np.zeros((powerset_size, n))

    logger.debug("Iterating over all subsets")
    utility_values = np.zeros(powerset_size)
    for i, subset in enumerate(
        maybe_progress(
            powerset(u.data.indices),
            progress,
            total=powerset_size - 1,
            position=0,
        )
    ):
        indices = np.zeros(n, dtype=bool)
        indices[list(subset)] = True
        A_lb[i, indices] = 1
        utility_values[i] = u(subset)

    b_lb = utility_values
    b_eq = utility_values[-1:]

    _, subsidy = _solve_least_core_linear_program(
        A_eq=A_eq, b_eq=b_eq, A_lb=A_lb, b_lb=b_lb, **options
    )

    values: Optional[NDArray[np.float_]]

    if subsidy is None:
        logger.debug("No values were found")
        status = ValuationStatus.Failed
        values = np.empty(n)
        values[:] = np.nan
        subsidy = np.nan

        return ValuationResult(
            algorithm="exact_least_core",
            status=status,
            values=values,
            subsidy=subsidy,
            stderr=None,
            data_names=u.data.data_names,
        )

    values = _solve_egalitarian_least_core_quadratic_program(
        subsidy,
        A_eq=A_eq,
        b_eq=b_eq,
        A_lb=A_lb,
        b_lb=b_lb,
        **options,
    )

    if values is None:
        logger.debug("No values were found")
        status = ValuationStatus.Failed
        values = np.empty(n)
        values[:] = np.nan
        subsidy = np.nan
    else:
        status = ValuationStatus.Converged

    return ValuationResult(
        algorithm="exact_least_core",
        status=status,
        values=values,
        subsidy=subsidy,
        stderr=None,
        data_names=u.data.data_names,
    )

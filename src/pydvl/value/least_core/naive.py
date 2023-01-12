import logging
import warnings
from typing import Optional

import numpy as np

from pydvl.utils import Utility, maybe_progress, powerset
from pydvl.value.least_core._common import _solve_linear_program
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
    :param options: LP Solver options. Refer to `SciPy's documentation
        <https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html>`_
        for more information
    :param progress: Whether to display a progress bar
    :return: Object with the data values.
    """
    n = len(u.data)

    # Arbitrary choice, will depend on time required, caching, etc.
    if n > 20:
        warnings.warn(f"Large dataset! Computation requires 2^{n} calls to model.fit()")

    if options is None:
        options = {}

    powerset_size = 2**n

    logger.debug("Building vectors and matrices for linear programming problem")
    c = np.zeros(n + 1)
    c[-1] = 1
    A_eq = np.ones((1, n + 1))
    A_eq[:, -1] = 0
    A_ub = np.zeros((powerset_size, n + 1))
    A_ub[:, -1] = -1

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
        indices = np.zeros(n + 1, dtype=bool)
        indices[list(subset)] = True
        A_ub[i, indices] = -1
        utility_values[i] = u(subset)

    b_ub = -utility_values
    b_eq = utility_values[-1:]

    values = _solve_linear_program(
        c,
        A_eq,
        b_eq,
        A_ub,
        b_ub,
        bounds=[(None, None)] * n + [(0.0, None)],
        **options,
    )

    if values is None:
        logger.debug("No values were found")
        status = ValuationStatus.Failed
        values = np.empty(n)
        values[:] = np.nan
    else:
        status = ValuationStatus.Converged

    # The last entry represents the least core value 'e'
    least_core_value = values[-1].item()
    values = values[:-1]

    return ValuationResult(
        algorithm="exact_least_core",
        status=status,
        values=values,
        stderr=None,
        data_names=u.data.data_names,
        least_core_value=least_core_value,
    )

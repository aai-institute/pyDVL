"""
.. versionadded:: 0.2.0
"""
import warnings

import numpy as np
import scipy

from ..reporting.scores import sort_values
from ..utils import Utility, maybe_progress, powerset

__all__ = ["naive_lc"]


def naive_lc(u: Utility, *, progress: bool = True, **kwargs):
    r"""


    .. math::

        \begin{array}{lll}
        \text{minimize}   & \displaystyle{e} & \\
        \text{subject to} & \displaystyle\sum_{i\in N} x_{i} = v(N) & \\
                          & \displaystyle\sum_{i\in S} x_{i} + e \geq v(S) & \forall S \subseteq N \\
        \end{array}
    """
    n = len(u.data)

    # Arbitrary choice, will depend on time required, caching, etc.
    if n > 20:
        warnings.warn(f"Large dataset! Computation requires 2^{n} calls to model.fit()")

    powerset_size = 2**n

    c = np.zeros(n + 1, dtype=np.int8)
    c[-1] = 1
    A_eq = np.ones((1, n + 1), dtype=np.int8)
    A_eq[:, -1] = 0
    A_ub = np.zeros((powerset_size, n + 1), dtype=np.int8)
    A_ub[:, -1] = -1

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
    result = scipy.optimize.linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        method="highs",
        bounds=(None, None),
    )

    if not result.success:
        warnings.warn("Could not find optimal solution", RuntimeWarning)

    values = result.x[:-1]
    sorted_values = sort_values({u.data.data_names[i]: v for i, v in enumerate(values)})

    return sorted_values

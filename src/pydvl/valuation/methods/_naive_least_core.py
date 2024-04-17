import logging
import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from pydvl.utils import powerset
from pydvl.valuation.methods._least_core_solving import (
    LeastCoreProblem,
    lc_solve_problem,
)
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.types import Sample
from pydvl.valuation.utility import Utility

__all__ = ["exact_least_core", "lc_prepare_problem"]

logger = logging.getLogger(__name__)


def exact_least_core(
    u: Utility,
    *,
    non_negative_subsidy: bool = False,
    solver_options: Optional[dict] = None,
    progress: bool = True,
) -> ValuationResult:
    r"""Computes the exact Least Core values.

    !!! Note
        If the training set contains more than 20 instances a warning is printed
        because the computation is very expensive. This method is mostly used for
        internal testing and simple use cases. Please refer to the
        [Monte Carlo method][pydvl.value.least_core.montecarlo.montecarlo_least_core]
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

    Args:
        u: Utility object with model, data, and scoring function
        non_negative_subsidy: If True, the least core subsidy $e$ is constrained
            to be non-negative.
        solver_options: Dictionary of options that will be used to select a solver
            and to configure it. Refer to the [cvxpy's
            documentation](https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options)
            for all possible options.
        progress: If True, shows a tqdm progress bar

    Returns:
        Object with the data values and the least core value.
    """
    n = len(u.training_data)
    if n > 20:  # Arbitrary choice, will depend on time required, caching, etc.
        warnings.warn(f"Large dataset! Computation requires 2^{n} calls to model.fit()")

    problem = lc_prepare_problem(u, progress=progress)
    return lc_solve_problem(
        problem=problem,
        u=u,
        algorithm="exact_least_core",
        non_negative_subsidy=non_negative_subsidy,
        solver_options=solver_options,
    )


def lc_prepare_problem(u: Utility, progress: bool = False) -> LeastCoreProblem:
    """Prepares a linear problem with all subsets of the data
    Use this to separate the problem preparation from the solving with
    [lc_solve_problem()][pydvl.value.least_core.common.lc_solve_problem]. Useful for
    parallel execution of multiple experiments.

    See [exact_least_core()][pydvl.value.least_core.naive.exact_least_core] for argument
    descriptions.
    """
    n = len(u.training_data)

    logger.debug("Building vectors and matrices for linear programming problem")
    powerset_size = 2**n
    A_lb = np.zeros((powerset_size, n))

    logger.debug("Iterating over all subsets")
    utility_values = np.zeros(powerset_size)
    for i, subset in enumerate(  # type: ignore
        tqdm(
            powerset(u.training_data.indices),
            disable=not progress,
            total=powerset_size - 1,
            position=0,
        )
    ):
        indices: NDArray[np.bool_] = np.zeros(n, dtype=bool)
        indices[list(subset)] = True
        A_lb[i, indices] = 1
        utility_values[i] = u(Sample(idx=0, subset=subset))  # type: ignore

    return LeastCoreProblem(utility_values, A_lb)

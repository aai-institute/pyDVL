"""
.. versionadded:: 0.4.0

This package holds all routines for the computation of Least Core data values.

Please refer to :ref:`data valuation` for an overview.

In addition to the standard interface via :func:`~pydvl.value.least_core.compute_least_core_values`,

"""
from enum import Enum
from typing import Optional

from .montecarlo import *
from .naive import *
from .. import ValuationResult
from ...utils import Utility


class LeastCoreMode(Enum):
    MonteCarlo = "montecarlo"
    Exact = "exact"


def compute_least_core_values(
    u: Utility,
    *,
    n_jobs: int = 1,
    n_iterations: Optional[int] = None,
    mode: LeastCoreMode = LeastCoreMode.MonteCarlo,
    **kwargs,
) -> ValuationResult:
    """Umbrella method to compute Least Core values with any of the available
    algorithms.

    See :ref:`data valuation` for an overview.

    The following algorithms are available. Note that the exact method can only
    work with very small datasets and is thus intended only for testing.

    - ``exact``: uses the complete powerset of the training set for the constraints
      :func:`~pydvl.value.shapley.naive.combinatorial_exact_shapley`.
    - ``montecarlo``:  uses the approximate Monte Carlo Least Core algorithm.
     Implemented in :func:`~pydvl.value.least_core.montecarlo.least_core_montecarlo`.
    """
    progress: bool = kwargs.pop("progress", False)

    if mode not in list(LeastCoreMode):
        raise ValueError(f"Invalid value encountered in {mode=}")

    if mode == LeastCoreMode.MonteCarlo:
        # TODO fix progress showing and maybe_progress in remote case
        progress = False
        if n_iterations is None:
            raise ValueError("n_iterations cannot be None for Monte Carlo Least Core")
        return montecarlo_least_core(
            u=u,
            n_iterations=n_iterations,
            n_jobs=n_jobs,
            progress=progress,
            **kwargs,
        )
    elif mode == LeastCoreMode.Exact:
        return exact_least_core(u=u, n_jobs=n_jobs, progress=progress, **kwargs)

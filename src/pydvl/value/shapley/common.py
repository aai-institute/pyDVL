from typing import Optional

from pydvl.utils import Utility
from pydvl.utils.types import Seed
from pydvl.value.result import ValuationResult
from pydvl.value.shapley.gt import group_testing_shapley
from pydvl.value.shapley.knn import knn_shapley
from pydvl.value.shapley.montecarlo import (
    combinatorial_montecarlo_shapley,
    permutation_montecarlo_shapley,
)
from pydvl.value.shapley.naive import (
    combinatorial_exact_shapley,
    permutation_exact_shapley,
)
from pydvl.value.shapley.owen import OwenAlgorithm, owen_sampling_shapley
from pydvl.value.shapley.truncated import NoTruncation
from pydvl.value.shapley.types import ShapleyMode
from pydvl.value.stopping import MaxChecks, StoppingCriterion

__all__ = ["compute_shapley_values"]


def compute_shapley_values(
    u: Utility,
    *,
    done: StoppingCriterion = MaxChecks(None),
    mode: ShapleyMode = ShapleyMode.TruncatedMontecarlo,
    n_jobs: int = 1,
    seed: Optional[Seed] = None,
    **kwargs,
) -> ValuationResult:
    """Umbrella method to compute Shapley values with any of the available
    algorithms.

    See [Data valuation][data-valuation-intro] for an overview.

    The following algorithms are available. Note that the exact methods can only
    work with very small datasets and are thus intended only for testing. Some
    algorithms also accept additional arguments, please refer to the
    documentation of each particular method.

    - `combinatorial_exact`: uses the combinatorial implementation of data
      Shapley. Implemented in
      [combinatorial_exact_shapley()][pydvl.value.shapley.naive.combinatorial_exact_shapley].
    - `combinatorial_montecarlo`:  uses the approximate Monte Carlo
      implementation of combinatorial data Shapley. Implemented in
      [combinatorial_montecarlo_shapley()][pydvl.value.shapley.montecarlo.combinatorial_montecarlo_shapley].
    - `permutation_exact`: uses the permutation-based implementation of data
      Shapley. Computation is **not parallelized**. Implemented in
      [permutation_exact_shapley()][pydvl.value.shapley.naive.permutation_exact_shapley].
    - `permutation_montecarlo`: uses the approximate Monte Carlo
      implementation of permutation data Shapley. Accepts a
      [TruncationPolicy][pydvl.value.shapley.truncated.TruncationPolicy] to stop
      computing marginals. Implemented in
      [permutation_montecarlo_shapley()][pydvl.value.shapley.montecarlo.permutation_montecarlo_shapley].
    - `owen_sampling`: Uses the Owen continuous extension of the utility
      function to the unit cube. Implemented in
      [owen_sampling_shapley()][pydvl.value.shapley.owen.owen_sampling_shapley]. This
      method does not take a [StoppingCriterion][pydvl.value.stopping.StoppingCriterion]
      but instead requires a parameter `q_max` for the number of subdivisions
      of the unit interval to use for integration, and another parameter
      `n_samples` for the number of subsets to sample for each $q$.
    - `owen_halved`: Same as 'owen_sampling' but uses correlated samples in the
      expectation. Implemented in
      [owen_sampling_shapley()][pydvl.value.shapley.owen.owen_sampling_shapley].
      This method  requires an additional parameter `q_max` for the number of
      subdivisions of the interval [0,0.5] to use for integration, and another
      parameter `n_samples` for the number of subsets to sample for each $q$.
    - `group_testing`: estimates differences of Shapley values and solves a
      constraint satisfaction problem. High sample complexity, not recommended.
      Implemented in [group_testing_shapley()][pydvl.value.shapley.gt.group_testing_shapley]. This
      method does not take a [StoppingCriterion][pydvl.value.stopping.StoppingCriterion]
      but instead requires a parameter `n_samples` for the number of
      iterations to run.

    Additionally, one can use model-specific methods:

    - `knn`: Exact method for K-Nearest neighbour models. Implemented in
      [knn_shapley()][pydvl.value.shapley.knn.knn_shapley].

    Args:
        u: [Utility][pydvl.utils.utility.Utility] object with model, data, and
            scoring function.
        done: Object used to determine when to stop the computation for Monte
            Carlo methods. The default is to stop after 100 iterations. See the
            available criteria in [stopping][pydvl.value.stopping]. It is
            possible to combine several of them using boolean operators. Some
            methods ignore this argument, others require specific subtypes.
        n_jobs: Number of parallel jobs (available only to some methods)
        seed: Either an instance of a numpy random number generator or a seed
            for it.
        mode: Choose which shapley algorithm to use. See
            [ShapleyMode][pydvl.value.shapley.ShapleyMode] for a list of allowed
            value.

    Returns:
        Object with the results.

    """
    progress: bool = kwargs.pop("progress", False)

    if mode not in list(ShapleyMode):
        raise ValueError(f"Invalid value encountered in {mode=}")

    if mode in (
        ShapleyMode.PermutationMontecarlo,
        ShapleyMode.ApproShapley,
        ShapleyMode.TruncatedMontecarlo,
    ):
        truncation = kwargs.pop("truncation", NoTruncation())
        return permutation_montecarlo_shapley(  # type: ignore
            u=u,
            done=done,
            truncation=truncation,
            n_jobs=n_jobs,
            seed=seed,
            progress=progress,
            **kwargs,
        )
    elif mode == ShapleyMode.CombinatorialMontecarlo:
        return combinatorial_montecarlo_shapley(  # type: ignore
            u, done=done, n_jobs=n_jobs, seed=seed, progress=progress
        )
    elif mode == ShapleyMode.CombinatorialExact:
        return combinatorial_exact_shapley(u, n_jobs=n_jobs, progress=progress)  # type: ignore
    elif mode == ShapleyMode.PermutationExact:
        return permutation_exact_shapley(u, progress=progress)
    elif mode == ShapleyMode.Owen or mode == ShapleyMode.OwenAntithetic:
        if kwargs.get("n_samples") is None:
            raise ValueError("n_samples cannot be None for Owen methods")
        if kwargs.get("max_q") is None:
            raise ValueError("Owen Sampling requires max_q for the outer integral")

        method = (
            OwenAlgorithm.Standard
            if mode == ShapleyMode.Owen
            else OwenAlgorithm.Antithetic
        )
        return owen_sampling_shapley(  # type: ignore
            u,
            n_samples=int(kwargs.get("n_samples", -1)),
            max_q=int(kwargs.get("max_q", -1)),
            method=method,
            n_jobs=n_jobs,
            seed=seed,
        )
    elif mode == ShapleyMode.KNN:
        return knn_shapley(u, progress=progress)
    elif mode == ShapleyMode.GroupTesting:
        n_samples = kwargs.pop("n_samples")
        if n_samples is None:
            raise ValueError("n_samples cannot be None for Group Testing")
        epsilon = kwargs.pop("epsilon")
        if epsilon is None:
            raise ValueError("Group Testing requires error bound epsilon")
        delta = kwargs.pop("delta", 0.05)
        return group_testing_shapley(  # type: ignore
            u,
            epsilon=float(epsilon),
            delta=delta,
            n_samples=int(n_samples),
            n_jobs=n_jobs,
            progress=progress,
            seed=seed,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid value encountered in {mode=}")

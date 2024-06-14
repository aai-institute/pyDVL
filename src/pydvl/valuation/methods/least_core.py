from __future__ import annotations

import math
import typing
from itertools import takewhile
from typing import Iterable, List

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from tqdm.auto import tqdm

from pydvl.utils.types import Seed
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods._solve_least_core_problems import (
    LeastCoreProblem,
    lc_solve_problem,
)
from pydvl.valuation.methods.gt_shapley import compute_utility_values_and_sample_masks
from pydvl.valuation.samplers.powerset import (
    DeterministicUniformSampler,
    NoIndexIteration,
    PowersetSampler,
    UniformSampler,
)
from pydvl.valuation.types import BatchGenerator, IndexSetT, SampleT
from pydvl.valuation.utility.base import UtilityBase

BoolDType = np.bool_

__all__ = [
    "LeastCoreValuation",
    "ExactLeastCoreValuation",
    "MonteCarloLeastCoreValuation",
]


class LeastCoreValuation(Valuation):
    """Umbrella class to calculate least-core values with multiple sampling methods.

    See [Data valuation][data-valuation] for an overview.

    Different samplers correspond to different least-core methods from the literature.
    For those, we provide convenience subclasses of LeastCoreValuation. See

    - [ExactLeastCoreValuation][pydvl.valuation.methods.least_core.ExactLeastCoreValuation]
    - [MonteCarloLeastCoreValuation][pydvl.valuation.methods.least_core.MonteCarloLeastCoreValuation]

    Other samplers allow you to create your own method and might yield computational
    gains over a standard Monte Carlo method.

    Args:
        utility: Utility object with model, data and scoring function.
        sampler: The sampler to use for the valuation.
        n_samples: The number of samples to use for the valuation. If None, it will be
            set to the sample limit of the chosen sampler (for finite samplers) or
            `1000 * len(data)` (for infinite samplers).
        non_negative_subsidy: If True, the least core subsidy $e$ is constrained
            to be non-negative.
        solver_options: Optional dictionary containing a CVXPY solver and options to
            configure it. For valid values to the "solver" key see
            [here](https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver).
            For additional options see [here](https://www.cvxpy.org/tutorial/solvers/index.html#setting-solver-options).
        progress: Whether to show a progress bar during the construction of the
            least-core problem.

    """

    def __init__(
        self,
        utility: UtilityBase,
        sampler: PowersetSampler,
        n_samples: int | None = None,
        non_negative_subsidy: bool = False,
        solver_options: dict | None = None,
        progress: bool = True,
    ):
        super().__init__()

        _check_sampler(sampler)
        self._utility = utility
        self._sampler = sampler
        self._non_negative_subsidy = non_negative_subsidy
        self._solver_options = solver_options
        self._n_samples = n_samples
        self._progress = progress

    def fit(self, data: Dataset) -> Valuation:
        """Calculate the least core valuation on a dataset.

        This method has to be called before calling `values()`.

        Calculating the least core valuation is a computationally expensive task that
        can be parallelized. To do so, call the `fit()` method inside a
        `joblib.parallel_config` context manager as follows:

        ```python
        from joblib import parallel_config

        with parallel_config(n_jobs=4):
            valuation.fit(data)
        ```

        """
        self._utility = self._utility.with_dataset(data)
        if self._n_samples is None:
            self._n_samples = _get_default_n_samples(
                sampler=self._sampler, indices=data.indices
            )

        algorithm = str(self._sampler)

        problem = create_least_core_problem(
            u=self._utility,
            sampler=self._sampler,
            n_samples=self._n_samples,
            progress=self._progress,
        )

        solution = lc_solve_problem(
            problem=problem,
            u=self._utility,
            algorithm=algorithm,
            non_negative_subsidy=self._non_negative_subsidy,
            solver_options=self._solver_options,
        )

        self.result = solution
        return self


class ExactLeastCoreValuation(LeastCoreValuation):
    """Class to calculate exact least-core values.

    Equivalent to calling `LeastCoreValuation` with a `DeterministicUniformSampler`
    and `n_samples=None`.

    The definition of the exact least-core valuation is:

    $$
    \begin{array}{lll}
    \text{minimize} & \displaystyle{e} & \\
    \text{subject to} & \displaystyle\sum_{i\in N} x_{i} = v(N) & \\
    & \displaystyle\sum_{i\in S} x_{i} + e \geq v(S) &, \forall S \subseteq N \\
    \end{array}
    $$

    Where $N = \{1, 2, \dots, n\}$ are the training set's indices.

    Args:
        utility: Utility object with model, data and scoring function.
        non_negative_subsidy: If True, the least core subsidy $e$ is constrained
            to be non-negative.
        solver_options: Optional dictionary containing a CVXPY solver and options to
            configure it. For valid values to the "solver" key see
            [here](https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver).
            For additional options see [here](https://www.cvxpy.org/tutorial/solvers/index.html#setting-solver-options).
        progress: Whether to show a progress bar during the construction of the
            least-core problem.

    """

    def __init__(
        self,
        utility: UtilityBase,
        non_negative_subsidy: bool = False,
        solver_options: dict | None = None,
        progress: bool = True,
    ):
        super().__init__(
            utility=utility,
            sampler=DeterministicUniformSampler(index_iteration=NoIndexIteration),
            n_samples=None,
            non_negative_subsidy=non_negative_subsidy,
            solver_options=solver_options,
            progress=progress,
        )


class MonteCarloLeastCoreValuation(LeastCoreValuation):
    """Class to calculate exact least-core values.

    Equivalent to calling `LeastCoreValuation` with a `UniformSampler`.

    The definition of the Monte Carlo least-core valuation is:

    $$
    \begin{array}{lll}
    \text{minimize} & \displaystyle{e} & \\
    \text{subject to} & \displaystyle\sum_{i\in N} x_{i} = v(N) & \\
    & \displaystyle\sum_{i\in S} x_{i} + e \geq v(S) & ,
    \forall S \in \{S_1, S_2, \dots, S_m \overset{\mathrm{iid}}{\sim} U(2^N) \}
    \end{array}
    $$

    Where:

    * $U(2^N)$ is the uniform distribution over the powerset of $N$.
    * $m$ is the number of subsets that will be sampled and whose utility will
      be computed and used to compute the data values.

    Args:
        utility: Utility object with model, data and scoring function.
        n_samples: The number of samples to use for the valuation. If None, it will be
            set to `1000 * len(data)`.
        non_negative_subsidy: If True, the least core subsidy $e$ is constrained
            to be non-negative.
        solver_options: Optional dictionary containing a CVXPY solver and options to
            configure it. For valid values to the "solver" key see
            [here](https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver).
            For additional options see [here](https://www.cvxpy.org/tutorial/solvers/index.html#setting-solver-options).
        progress: Whether to show a progress bar during the construction of the
            least-core problem.


    """

    def __init__(
        self,
        utility: UtilityBase,
        n_samples: int,
        non_negative_subsidy: bool = False,
        solver_options: dict | None = None,
        progress: bool = True,
        seed: Seed | None = None,
    ):
        super().__init__(
            utility=utility,
            sampler=UniformSampler(index_iteration=NoIndexIteration, seed=seed),
            n_samples=n_samples,
            non_negative_subsidy=non_negative_subsidy,
            solver_options=solver_options,
            progress=progress,
        )


def create_least_core_problem(
    u: UtilityBase, sampler: PowersetSampler, n_samples: int, progress: bool
) -> LeastCoreProblem:
    """Create a Least Core problem from a utility and a sampler.

    Args:
        u: Utility object with model, data and scoring function.
        sampler: The sampler to use for the valuation.
        n_samples: The maximum number of samples to use for the valuation.
        progress: Whether to show a progress bar during the construction of the
            least-core problem.

    Returns:
        LeastCoreProblem: The least core problem to solve.

    """
    utility_values, masks = compute_utility_values_and_sample_masks(
        utility=u, sampler=sampler, n_samples=n_samples, progress=progress
    )

    return LeastCoreProblem(utility_values=utility_values, A_lb=masks.astype(float))


def _get_default_n_samples(sampler: PowersetSampler, indices: IndexSetT) -> int:
    """Get a default value for n_samples based on the sampler's sample limit.

    Args:
        sampler: The sampler to use for the valuation.
        indices: The indices of the dataset.

    Returns:
        int: The number of samples to use for the valuation.

    """
    sample_limit = sampler.sample_limit(indices)
    if sample_limit is not None:
        out = sample_limit
    else:
        # TODO: This is a rather arbitrary rule of thumb. The value was chosen to be
        # larger than the number of samples used by Yan and Procaccia
        # https://ojs.aaai.org/index.php/AAAI/article/view/16721 in a low resource
        # setting but linear in the dataset size to avoid exploding runtimes.
        out = 1000 * len(indices)

    return out


def _check_sampler(sampler: PowersetSampler):
    """Check that the sampler is compatible with the Least Core valuation."""
    if sampler._index_iteration != NoIndexIteration:
        raise ValueError(
            "Least core valuation only supports samplers with NoIndexIteration."
        )

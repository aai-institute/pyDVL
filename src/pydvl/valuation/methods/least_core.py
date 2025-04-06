"""
This module implements the least-core valuation method.

Least-Core values were introduced by Yan and Procaccia (2021).[^1] Please refer to the
paper or our [documentation][least-core-intro] for more details and a [comparison with
other methods][least-core-comparison] (Benmerzoug and de Benito Delgado, 2023).[^2]

## References

[^1]: Yan, Tom, and Ariel D. Procaccia. [If You Like Shapley Then You’ll Love the
      Core](https://doi.org/10.1609/aaai.v35i6.16721). In Proceedings of the 35th AAAI
      Conference on Artificial Intelligence, 2021, 6:5751–59. Virtual conference:
      Association for the Advancement of Artificial Intelligence, 2021.
[^2]: Benmerzoug, Anes, and Miguel de Benito Delgado. [[Re] If You like Shapley, Then
      You’ll Love the Core](https://doi.org/10.5281/zenodo.8173733). ReScience C 9, no.
      2 (31 July 2023): #32.
"""

from __future__ import annotations

import numpy as np
from typing_extensions import Self

from pydvl.utils.types import Seed
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods._solve_least_core_problems import (
    LeastCoreProblem,
    lc_solve_problem,
)
from pydvl.valuation.methods._utility_values_and_sample_masks import (
    compute_utility_values_and_sample_masks,
)
from pydvl.valuation.samplers.powerset import (
    DeterministicUniformSampler,
    FiniteNoIndexIteration,
    NoIndexIteration,
    PowersetSampler,
    UniformSampler,
)
from pydvl.valuation.types import IndexSetT
from pydvl.valuation.utility.base import UtilityBase

BoolDType = np.bool_

__all__ = [
    "LeastCoreValuation",
    "ExactLeastCoreValuation",
    "MonteCarloLeastCoreValuation",
]


class LeastCoreValuation(Valuation):
    """Umbrella class to calculate least-core values with multiple sampling methods.

    See [the documentation][least-core-intro] for an overview.

    Different samplers correspond to different least-core methods from the literature.
    For those, we provide convenience subclasses of `LeastCoreValuation`. See

    - [ExactLeastCoreValuation][pydvl.valuation.methods.least_core.ExactLeastCoreValuation]
    - [MonteCarloLeastCoreValuation][pydvl.valuation.methods.least_core.MonteCarloLeastCoreValuation]

    Other samplers allow you to create your own importance sampling method and might
    yield computational gains over the standard Monte Carlo method.

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
            For additional options see
            [here](https://www.cvxpy.org/tutorial/solvers/index.html#setting-solver-options).
        progress: Whether to show a progress bar during the construction of the
            least-core problem.

    """

    algorithm_name = "Least-Core"

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
        self.algorithm_name = f"LeastCore-{str(sampler)}"

    def fit(self, data: Dataset) -> Self:
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

        problem = create_least_core_problem(
            u=self._utility,
            sampler=self._sampler,
            n_samples=self._n_samples,
            progress=self._progress,
        )

        solution = lc_solve_problem(
            problem=problem,
            u=self._utility,
            algorithm=str(self),
            non_negative_subsidy=self._non_negative_subsidy,
            solver_options=self._solver_options,
        )

        self.result = self.init_or_check_result(data)
        self.result += solution
        return self


class ExactLeastCoreValuation(LeastCoreValuation):
    """Class to calculate exact least-core values.

    Equivalent to constructing a
    [LeastCoreValuation][pydvl.valuation.methods.least_core.LeastCoreValuation] with a
    [DeterministicUniformSampler][pydvl.valuation.samplers.powerset.DeterministicUniformSampler]
    and `n_samples=None`.

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

    algorithm_name = "Exact-Least-Core"

    def __init__(
        self,
        utility: UtilityBase,
        non_negative_subsidy: bool = False,
        solver_options: dict | None = None,
        progress: bool = True,
        batch_size: int = 1,
    ):
        super().__init__(
            utility=utility,
            sampler=DeterministicUniformSampler(
                index_iteration=FiniteNoIndexIteration, batch_size=batch_size
            ),
            n_samples=None,
            non_negative_subsidy=non_negative_subsidy,
            solver_options=solver_options,
            progress=progress,
        )


class MonteCarloLeastCoreValuation(LeastCoreValuation):
    """Class to calculate exact least-core values.

    Equivalent to creating a
    [LeastCoreValuation][pydvl.valuation.methods.least_core.LeastCoreValuation]
    with a [UniformSampler][pydvl.valuation.samplers.powerset.UniformSampler].

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

    algorithm_name = "Monte-Carlo-Least-Core"

    def __init__(
        self,
        utility: UtilityBase,
        n_samples: int,
        non_negative_subsidy: bool = False,
        solver_options: dict | None = None,
        progress: bool = True,
        seed: Seed | None = None,
        batch_size: int = 1,
    ):
        super().__init__(
            utility=utility,
            sampler=UniformSampler(
                index_iteration=NoIndexIteration, seed=seed, batch_size=batch_size
            ),
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
    if not issubclass(sampler._index_iterator_cls, NoIndexIteration):
        raise ValueError(
            "Least core valuation only supports samplers with NoIndexIteration."
        )

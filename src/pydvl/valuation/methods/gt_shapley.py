"""
This module implements Group Testing for the approximation of Shapley values, as
introduced in (Jia, R. et al., 2019).[^1]

Computation of the Shapley value is transformed into a feasibility problem, where the
constraints involve certain utility evaluations of sampled subsets. The sampling is done
in such a way that an approximation to the true Shapley values can be computed with
guarantees.[^3]

!!! Warning
    Group Testing is very inefficient. Potential improvements to the implementation
    notwithstanding, convergence seems to be very slow (in terms of evaluations of the
    utility required). We recommend other Monte Carlo methods instead. See [the whole
    list here][implemented-methods-data-valuation].

You can read about data valuation in general [in the documentation][data-valuation-intro].

## References

[^1]: <a name="jia_efficient_2019"></a>Jia, R. et al., 2019.
      [Towards Efficient Data Valuation Based on the Shapley
      Value](https://proceedings.mlr.press/v89/jia19a.html).
      In: Proceedings of the 22nd International Conference on Artificial Intelligence
      and Statistics, pp. 1167–1176. PMLR.
[^2]: <a name="jia_update_2023"></a>Jia, R. et al., 2023.
    [A Note on "Towards Efficient Data Valuation Based on the Shapley
    Value"](https://arxiv.org/pdf/2302.11431).
[^3]: Internally, this sampling is achieved with a
      [StratifiedSampler][pydvl.valuation.samplers.StratifiedSampler] with
      [GroupTestingSampleSize][pydvl.valuation.samplers.GroupTestingSampleSize] strategy.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, cast

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from pydvl.utils.status import Status
from pydvl.utils.types import Seed
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods._utility_values_and_sample_masks import (
    compute_utility_values_and_sample_masks,
)
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers import (
    GroupTestingSampleSize,
    IndexSampler,
    NoIndexIteration,
    RandomSizeIteration,
    StratifiedSampler,
)
from pydvl.valuation.types import NameT, Sample
from pydvl.valuation.utility.base import UtilityBase

log = logging.getLogger(__name__)

__all__ = ["GroupTestingShapleyValuation", "compute_n_samples"]


class GroupTestingShapleyValuation(Valuation):
    """Class to calculate the group-testing approximation to shapley values.

    See (Jia, R. et al., 2019)<sup><a href="#jia_update_2023">1</a></sup> for a
    description of the method, and [Data valuation][data-valuation-intro] for an overview of
    data valuation.

    !!! Warning
        Group Testing is very inefficient. Potential improvements to the implementation
        notwithstanding, convergence seems to be very slow (in terms of evaluations of
        the utility required). We recommend other Monte Carlo methods instead. See [the
        whole list here][implemented-methods-data-valuation].

    Args:
        utility: Utility object with model, data and scoring function.
        n_samples: The number of samples to use. A sample size with theoretical
            guarantees can be computed using
            [compute_n_samples()][pydvl.valuation.methods.gt_shapley.compute_n_samples].
        epsilon: The error tolerance.
        solver_options: Optional dictionary containing a CVXPY solver and options to
            configure it. For valid values to the "solver" key see [this
            tutorial](https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver).
            For additional options [cvxpy's
            documentation](https://www.cvxpy.org/tutorial/solvers/index.html#setting-solver-options).
        progress: Whether to show a progress bar during the construction of the
            group-testing problem.
        seed: Seed for the random number generator.
        batch_size: The number of samples to draw in each batch. Can be used to reduce
            parallelization overhead for fast utilities. Defaults to 1.
    """

    algorithm_name = "Group-Testing-Shapley"

    def __init__(
        self,
        utility: UtilityBase,
        n_samples: int,
        epsilon: float,
        solver_options: dict | None = None,
        progress: bool = True,
        seed: Seed | None = None,
        batch_size: int = 1,
    ):
        super().__init__()

        self._utility = utility
        self._n_samples = n_samples
        self._solver_options = solver_options
        self._progress = progress
        self._sampler = StratifiedSampler(
            index_iteration=NoIndexIteration,
            sample_sizes=GroupTestingSampleSize(),
            sample_sizes_iteration=RandomSizeIteration,
            batch_size=batch_size,
            seed=seed,
        )
        self._epsilon = epsilon

    def fit(self, data: Dataset, continue_from: ValuationResult | None = None) -> Self:
        """Calculate the group-testing valuation on a dataset.

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
        self._result = self._init_or_check_result(data, continue_from)
        self._utility = self._utility.with_dataset(data)

        problem = create_group_testing_problem(
            utility=self._utility,
            sampler=self._sampler,
            n_samples=self._n_samples,
            progress=self._progress,
            epsilon=self._epsilon,
        )

        solution = solve_group_testing_problem(
            problem=problem,
            solver_options=self._solver_options,
            algorithm_name=self.algorithm_name,
            data_names=data.names,
        )

        self._result += solution
        return self


def compute_n_samples(epsilon: float, delta: float, n_obs: int) -> int:
    r"""Compute the minimal sample size with epsilon-delta guarantees.

    Based on the formula in Theorem 4 of
    (Jia, R. et al., 2023)<sup><a href="#jia_update_2023">2</a></sup>
    which gives a lower bound on the number of samples required to obtain an
    (ε/√n,δ/(N(N-1))-approximation to all pair-wise differences of Shapley
    values, wrt. $\ell_2$ norm.

    The updated version refines the lower bound of the original paper. Note that the
    bound is tighter than earlier versions but might still overestimate the number of
    samples required.

    Args:
        epsilon: The error tolerance.
        delta: The confidence level.
        n_obs: Number of data points.

    Returns:
        The sample size.

    """
    kk = _create_sample_sizes(n_obs)
    Z = _calculate_z(n_obs)

    q = _create_sampling_probabilities(kk)
    q_tot = (n_obs - 2) / n_obs * q[0] + np.inner(
        q[1:], 1 + 2 * kk[1:] * (kk[1:] - n_obs) / (n_obs * (n_obs - 1))
    )

    def _h(u: float) -> float:
        return cast(float, (1 + u) * np.log(1 + u) - u)

    n_samples = np.log(n_obs * (n_obs - 1) / delta)
    n_samples /= 1 - q_tot
    n_samples /= _h(epsilon / (2 * Z * np.sqrt(n_obs) * (1 - q_tot)))

    return int(n_samples)


class GroupTestingProblem(NamedTuple):
    """Solver agnostic representation of the group-testing problem."""

    utility_differences: NDArray[np.float64]
    total_utility: float
    epsilon: float


def create_group_testing_problem(
    utility: UtilityBase,
    sampler: IndexSampler,
    n_samples: int,
    progress: bool,
    epsilon: float,
) -> GroupTestingProblem:
    """Create the feasibility problem that characterizes group testing shapley values.

    Args:
        utility: Utility object with model, data and scoring function.
        sampler: The sampler to use for the valuation.
        n_samples: The number of samples to use.
        progress: Whether to show a progress bar.
        epsilon: The error tolerance.

    Returns:
        A GroupTestingProblem object.

    """
    if utility.training_data is None:
        raise ValueError("Utility object must have training data.")

    u_values, masks = compute_utility_values_and_sample_masks(
        utility=utility,
        sampler=sampler,
        n_samples=n_samples,
        progress=progress,
        extra_samples=[Sample(idx=None, subset=utility.training_data.indices)],
    )

    total_utility = u_values[-1].item()
    u_values = u_values[:-1]
    masks = masks[:-1]

    u_differences = _calculate_utility_differences(
        utility_values=u_values, masks=masks, n_obs=len(utility.training_data)
    )

    problem = GroupTestingProblem(
        utility_differences=u_differences,
        total_utility=total_utility,
        epsilon=epsilon,
    )

    return problem


def _calculate_utility_differences(
    utility_values: NDArray[np.float64],
    masks: NDArray[np.bool_],
    n_obs: int,
) -> NDArray[np.float64]:
    """Calculate utility differences from utility values and sample masks.

    Args:
        utility_values: 1D array with utility values.
        masks: 2D array with sample masks.
        n_obs: The number of observations.

    Returns:
        The utility differences.

    """
    betas: NDArray[np.int_] = masks.astype(np.int_)
    n_samples = len(utility_values)

    z = _calculate_z(n_obs)

    u_differences = np.zeros(shape=(n_obs, n_obs))
    for i in range(n_obs):
        for j in range(i + 1, n_obs):
            u_differences[i, j] = np.dot(utility_values, betas[:, i] - betas[:, j])
    u_differences *= z / n_samples

    return u_differences


def solve_group_testing_problem(
    problem: GroupTestingProblem,
    solver_options: dict | None,
    algorithm_name: str,
    data_names: NDArray[NameT],
) -> ValuationResult:
    """Solve the group testing problem and create a ValuationResult.

    Args:
        problem: The group testing problem.
        solver_options: Optional dictionary containing a CVXPY solver and options to
            configure it. For valid values to the "solver" key see
            [here](https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver).
            For additional options see
            [here](https://www.cvxpy.org/tutorial/solvers/index.html#setting-solver-options).
        algorithm_name: The name of the algorithm.
        data_names: The names of the data columns.

    Returns:
        A ValuationResult object.

    """

    solver_options = {} if solver_options is None else solver_options.copy()

    C = problem.utility_differences
    total_utility = problem.total_utility
    epsilon = problem.epsilon
    n_obs = len(C)

    v = cp.Variable(n_obs)
    constraints = [cp.sum(v) == total_utility]
    for i in range(n_obs):
        for j in range(i + 1, n_obs):
            constraints.append(v[i] - v[j] <= epsilon + C[i, j])
            constraints.append(v[j] - v[i] <= epsilon - C[i, j])

    cp_problem = cp.Problem(cp.Minimize(0), constraints)
    solver = solver_options.pop("solver", cp.SCS)
    cp_problem.solve(solver=solver, **solver_options)

    if cp_problem.status != "optimal":
        log.warning(f"cvxpy returned status {cp_problem.status}")
        values = (
            np.nan * np.ones_like(n_obs)
            if not hasattr(v.value, "__len__")
            else cast(NDArray[np.float64], v.value)
        )
        status = Status.Failed
    else:
        values = cast(NDArray[np.float64], v.value)
        status = Status.Converged

    result = ValuationResult(
        status=status,
        values=values,
        data_names=data_names,
        solver_status=cp_problem.status,
        algorithm=algorithm_name,
    )

    return result


def _create_sample_sizes(n_obs: int) -> NDArray[np.int_]:
    """Create a grid of possible sample sizes for the group testing algorithm."""
    return np.arange(1, n_obs)


def _create_sampling_probabilities(
    sample_sizes: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Create probabilities for each possible sample size."""
    weights = 1 / sample_sizes + 1 / sample_sizes[::-1]
    probs: NDArray[np.float64] = weights / weights.sum()
    return probs


def _calculate_z(n_obs: int) -> float:
    """Calculate the normalization constant Z."""
    kk = _create_sample_sizes(n_obs)
    z: float = 2 * np.sum(1 / kk)
    return z

"""
This module implements Group Testing for the approximation of Shapley values, as
introduced in (Jia, R. et al., 2019)[^1]. The sampling of index subsets is
done in such a way that an approximation to the true Shapley values can be
computed with guarantees.

!!! Warning
    This method is very inefficient. Potential improvements to the
    implementation notwithstanding, convergence seems to be very slow (in terms
    of evaluations of the utility required). We recommend other Monte Carlo
    methods instead.

You can read more [in the documentation][data-valuation].

!!! tip "New in version 0.4.0"

## References

[^1]: <a name="jia_efficient_2019"></a>Jia, R. et al., 2019.
    [Towards Efficient Data Valuation Based on the Shapley
    Value](https://proceedings.mlr.press/v89/jia19a.html).
    In: Proceedings of the 22nd International Conference on Artificial
    Intelligence and Statistics, pp. 1167–1176. PMLR.
[^2]: <a name="jia_update_2023"></a>Jia, R. et al., 2023.
    [A Note on "Towards Efficient Data Valuation Based on the Shapley Value"](
    https://arxiv.org/pdf/2302.11431).

"""
from __future__ import annotations

import logging
import math
from itertools import chain, takewhile
from typing import Iterable, NamedTuple, Sequence, Tuple, cast

import cvxpy as cp
import numpy as np
from joblib import Parallel, delayed
from more_itertools import batched
from numpy.typing import NDArray
from tqdm.auto import tqdm
from typing_extensions import Self

from pydvl.utils.numeric import random_subset_of_size
from pydvl.utils.status import Status
from pydvl.utils.types import Seed
from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.samplers.base import IndexSampler
from pydvl.valuation.samplers.utils import StochasticSamplerMixin
from pydvl.valuation.types import BatchGenerator, IndexSetT, Sample, SampleGenerator
from pydvl.valuation.utility.base import UtilityBase

log = logging.getLogger(__name__)

__all__ = ["GroupTestingValuation", "compute_n_samples"]


class GroupTestingValuation(Valuation):
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
        self._sampler = GTSampler(batch_size=batch_size, seed=seed)
        self._epsilon = epsilon

    def fit(self, data: Dataset) -> Self:
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
        self._utility = self._utility.with_dataset(data)

        problem = _create_group_testing_problem(
            utility=self._utility,
            sampler=self._sampler,
            n_samples=self._n_samples,
            progress=self._progress,
            epsilon=self._epsilon,
        )

        solution = _solve_group_testing_problem(
            problem=problem,
            solver_options=self._solver_options,
            algorithm_name=self.algorithm_name,
            data_names=data.data_names,
        )

        self.result = solution
        return self


def compute_n_samples(epsilon: float, delta: float, n_obs: int) -> int:
    """Compute the minimal sample size with epsilon-delta guarantees.

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
        return (1 + u) * np.log(1 + u) - u

    n_samples = np.log(n_obs * (n_obs - 1) / delta)
    n_samples /= 1 - q_tot
    n_samples /= _h(epsilon / (2 * Z * np.sqrt(n_obs) * (1 - q_tot)))

    return int(n_samples)


class GroupTestingProblem(NamedTuple):
    """Solver agnostic representation of the group-testing problem."""

    utility_differences: NDArray[np.float_]
    total_utility: float
    epsilon: float


class GTSampler(StochasticSamplerMixin, IndexSampler):
    def __init__(self, batch_size: int = 1, seed: Seed | None = None):
        super().__init__(batch_size=batch_size, seed=seed)

    def _generate(self, indices: IndexSetT) -> SampleGenerator:
        n_obs = len(indices)
        sample_sizes = _create_sample_sizes(n_obs)
        probabilities = _create_sampling_probabilities(sample_sizes)

        while True:
            size = self._rng.choice(sample_sizes, p=probabilities, size=1)
            subset = random_subset_of_size(indices, size=size, seed=self._rng)
            yield Sample(idx=None, subset=subset)

    def weight(n: int, subset_len: int) -> float:
        raise NotImplementedError("This is not a semi-value sampler.")

    def make_strategy(
        self,
        utility: UtilityBase,
        coefficient: Callable[[int, int], float] | None = None,
    ) -> EvaluationStrategy:
        raise NotImplementedError("This is not a semi-value sampler.")


def compute_utility_values_and_sample_masks(
    utility: UtilityBase,
    sampler: GTSampler,
    n_samples: int,
    progress: bool,
    extra_samples: Iterable[SampleT] | None = None,
) -> Tuple[NDArray[np.float_], NDArray[np.int_]]:
    """Calculate utility values and sample masks on samples in parallel.

    Creating the utility evaluations and sample masks is the computational bottleneck
    of several data valuation algorithms, for examples least-core and group-testing.

    """
    if utility.training_data is None:
        raise ValueError("Utility object must have training data.")

    indices = utility.training_data.indices
    n_obs = len(indices)

    batch_size = sampler.batch_size
    n_batches = math.ceil(n_samples / batch_size)

    def _create_mask_and_utility_values(
        batch: Iterable[SampleT],
    ) -> tuple[List[NDArray[BoolDType]], List[float]]:
        """Convert sampled indices to boolean masks and calculate utility on each
        sample in batch."""
        masks: List[NDArray[BoolDType]] = []
        u_values: List[float] = []
        for sample in batch:
            m = np.full(n_obs, False)
            m[sample.subset.astype(int)] = True
            masks.append(m)
            u_values.append(utility(sample))

        return masks, u_values

    generator = takewhile(
        lambda _: sampler.n_samples < n_samples,
        sampler.generate_batches(indices),
    )

    if extra_samples is not None:
        generator = chain(generator, batched(extra_samples, batch_size))

    generator_with_progress = cast(
        BatchGenerator,
        tqdm(
            generator,
            disable=not progress,
            total=n_batches - 1,
            position=0,
        ),
    )

    parallel = Parallel(return_as="generator")

    results = parallel(
        delayed(_create_mask_and_utility_values)(batch)
        for batch in generator_with_progress
    )

    masks: List[NDArray[BoolDType]] = []
    u_values: List[float] = []
    for m, v in results:
        masks.extend(m)
        u_values.extend(v)

    u_values = np.array(u_values)
    masks = np.row_stack(masks)

    return u_values, masks


def _create_group_testing_problem(utility, sampler, n_samples, progress, epsilon):
    u_values, masks = compute_utility_values_and_sample_masks(
        utility=utility,
        sampler=sampler,
        n_samples=n_samples,
        progress=progress,
        extra_samples=[Sample(idx=None, subset=utility.training_data.indices)],
    )

    total_utility = u_values[-1]
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


def _calculate_utility_differences(utility_values, masks, n_obs):
    betas = masks.astype(np.int_)
    n_samples = len(utility_values)

    z = _calculate_z(n_obs)

    u_differences = np.zeros(shape=(n_obs, n_obs))
    for i in range(n_obs):
        for j in range(i + 1, n_obs):
            u_differences[i, j] = np.dot(utility_values, betas[:, i] - betas[:, j])
    u_differences *= z / n_samples

    return u_differences


def _solve_group_testing_problem(
    problem: GroupTestingProblem,
    solver_options: dict | None,
    algorithm_name: str = "",
    data_names: Sequence[str] | None = None,
) -> ValuationResult:
    """Solve the group testing problem and create a ValuationResult."""

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
            np.nan * np.ones_like(u.data.indices)
            if not hasattr(v.value, "__len__")
            else v.value
        )
        status = Status.Failed
    else:
        values = v.value
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
) -> NDArray[np.float_]:
    """Create probabilities for each possible sample size."""
    weights = 1 / sample_sizes + 1 / sample_sizes[::-1]
    probs = weights / weights.sum()
    return probs


def _calculate_z(n_obs):
    kk = _create_sample_sizes(n_obs)
    return 2 * np.sum(1 / kk)

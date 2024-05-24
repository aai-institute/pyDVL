from __future__ import annotations

import math
import typing
import warnings
from itertools import takewhile
from typing import Iterable, List

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from tqdm.auto import tqdm

from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods._solve_least_core_problems import (
    LeastCoreProblem,
    lc_solve_problem,
)
from pydvl.valuation.samplers.powerset import NoIndexIteration, PowersetSampler
from pydvl.valuation.types import BatchGenerator, SampleT
from pydvl.valuation.utility.base import UtilityBase

BoolDType = np.bool_

__all__ = ["LeastCoreValuation"]


class LeastCoreValuation(Valuation):
    """Umbrella class to calculate Least Core values with multiple sampling methods.

    See [Data valuation][data-valuation] for an overview.

    Different samplers correspond to different least-core methods from the literature:

    - `DeterministicUniformSampler`: Exact (naive) method. This yields the most precise
        results but is only feasible for tiny datasets (<= 20 observations).
    - `UniformSampler`: Monte Carlo method. This is the most practical method for
        larger datasets.

    Other samplers allow you to create your own method and might yield computational
    gains over a standard Monte Carlo method.

    Args:
        utility: Utility object with model, data and scoring function.
        sampler: The sampler to use for the valuation.
        n_samples: The number of samples to use for the valuation. Can be set
            to None if deterministic samplers with known number of samples (e.g.
            DeterministicUniformSampler) are used.
        If True, the least core subsidy $e$ is constrained
            to be non-negative.
        solver_options: Optional dictionary of options passed to the solvers.
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

        self._n_samples = _correct_n_samples(
            candidate=self._n_samples,
            sampler_length=self._sampler.sample_limit(data.indices),
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
    if u.training_data is not None:
        n_indices = len(u.training_data.indices)
    else:
        raise ValueError("Utility object must have a training dataset.")

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
            m = np.full(n_indices, False)
            m[sample.subset.astype(int)] = True
            masks.append(m)
            u_values.append(u(sample))

        return masks, u_values

    generator = takewhile(
        lambda _: sampler.n_samples < n_samples,
        sampler.generate_batches(u.training_data.indices),
    )

    generator_with_progress = typing.cast(
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

    utility_values = np.array(u_values)
    A_lb = np.row_stack(masks).astype(float)

    return LeastCoreProblem(utility_values=utility_values, A_lb=A_lb)


def _correct_n_samples(candidate: int | None, sampler_length: int | None) -> int:
    """Correct a user provided n_samples parameter

    Args:
        candidate: The user provided value for n_samples.
        sampler_length: The length of the sampler which is None for infinite samplers.

    Returns:
        int: The number of samples to use for the valuation.

    """
    if sampler_length is not None:
        if candidate is not None and candidate != sampler_length:
            warnings.warn(
                f"Invalid value for n_samples: {candidate}. Setting to {sampler_length}."
            )
        out = sampler_length

        if sampler_length >= 2**20:
            warnings.warn(
                "PerformanceWarning: Your combination of sampler and dataset size may "
                "lead to slow performance. Consider using randomized samplers."
            )
    else:
        if candidate is None:
            raise ValueError(
                "n_samples must be set if a sampler with infinite length is used."
            )
        out = candidate

    return out


def _check_sampler(sampler: PowersetSampler):
    """Check that the sampler is compatible with the Least Core valuation."""
    if sampler._index_iteration != NoIndexIteration:
        raise ValueError(
            "Least core valuation only supports samplers with NoIndexIteration."
        )

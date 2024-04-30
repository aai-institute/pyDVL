from __future__ import annotations

import warnings
from itertools import islice

import numpy as np
from tqdm.auto import tqdm

from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods._least_core_solving import (
    LeastCoreProblem,
    lc_solve_problem,
)
from pydvl.valuation.samplers.base import IndexSampler
from pydvl.valuation.samplers.powerset import NoIndexIteration
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["LeastCoreValuation"]


class LeastCoreValuation(Valuation):
    def __init__(
        self,
        utility: UtilityBase,
        sampler: IndexSampler,
        max_samples: int | None = None,
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
        self._max_samples = max_samples
        self._progress = progress

    def fit(self, data: Dataset) -> None:

        self._utility = self._utility.with_dataset(data)

        self._max_samples = _process_max_samples(
            candidate=self._max_samples,
            sampler_length=self._sampler.length(data.indices),
        )

        # ==============================================================================
        # Things that should not exist
        algorithm = "placeholder"
        # ==============================================================================

        problem = create_least_core_problem(
            u=self._utility,
            sampler=self._sampler,
            max_samples=self._max_samples,
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


def create_least_core_problem(
    u: UtilityBase, sampler: IndexSampler, max_samples: int, progress: bool
):
    n_obs = len(u.training_data)

    A_lb = np.zeros((max_samples, n_obs))
    utility_values = np.zeros(max_samples)

    generator = sampler.from_indices(u.training_data.indices)
    for i, batch in enumerate(  # type: ignore
        tqdm(
            islice(generator, max_samples),
            disable=not progress,
            total=max_samples - 1,
            position=0,
        )
    ):
        sample = list(batch)[0]
        A_lb[i, sample.subset.astype(int)] = 1
        utility_values[i] = u(sample)

    return LeastCoreProblem(utility_values=utility_values, A_lb=A_lb)


def _process_max_samples(
    candidate: int | None, sampler_length: int | None
) -> int | None:
    if sampler_length is not None:
        if candidate is not None and candidate != sampler_length:
            warnings.warn(
                f"Invalid value for max_samples: {candidate}. Setting to {sampler_length}."
            )
        out = sampler_length

        if sampler_length >= 2**20:
            warnings.warn(
                "PerformanceWarning: Your combination of sampler and dataset size may "
                "lead to slow performance. Consider using randomized samplers."
            )
    else:
        out = candidate

    return out


def _check_sampler(sampler: IndexSampler) -> IndexSampler:
    if sampler.batch_size != 1:
        raise ValueError("Least core valuation only supports batch_size=1 samplers.")
    if sampler._index_iteration != NoIndexIteration:
        raise ValueError(
            "Least core valuation only supports samplers with NoIndexIteration."
        )

from __future__ import annotations

import warnings
from enum import Enum
from itertools import islice

import numpy as np

from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods._least_core_solving import (
    LeastCoreProblem,
    lc_solve_problem,
)
from pydvl.valuation.methods._montecarlo_least_core import montecarlo_least_core
from pydvl.valuation.methods._naive_least_core import exact_least_core
from pydvl.valuation.samplers import DeterministicUniformSampler
from pydvl.valuation.samplers.base import IndexSampler
from pydvl.valuation.utility import Utility
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["LeastCoreValuation"]


class LeastCoreMode(Enum):
    """Available Least Core algorithms."""

    MonteCarlo = "montecarlo"
    Exact = "exact"


class LeastCoreValuation(Valuation):
    def __init__(
        self,
        utility: UtilityBase,
        sampler: IndexSampler,
        n_iterations: int | None = None,
        non_negative_subsidy: bool = False,
        solver_options: dict | None = None,
    ):
        super().__init__()
        self._utility = utility
        self._sampler = sampler
        self._non_negative_subsidy = non_negative_subsidy
        self._solver_options = solver_options
        self._n_iterations = n_iterations

    def fit(self, data: Dataset) -> None:

        self._utility = self._utility.with_dataset(data)

        # ==============================================================================
        # Things that should not exist
        algorithm = "placeholder"

        if isinstance(self._sampler, DeterministicUniformSampler):
            _correct_n_iterations = 2 ** len(self._utility.training_data)
            if self._n_iterations != _correct_n_iterations:
                warnings.warn(
                    "Incorrect number of iterations for exact least core: "
                    f"{n_iterations}. Setting to {_correct_n_iterations}."
                )
                self._n_iterations = _correct_n_iterations

        # ==============================================================================

        problem = create_least_core_problem(
            u=self._utility, sampler=self._sampler, n_iterations=self._n_iterations
        )

        solution = lc_solve_problem(
            problem=problem,
            u=self._utility,
            algorithm=algorithm,
            non_negative_subsidy=self._non_negative_subsidy,
            solver_options=self._solver_options,
        )

        self.result = solution


def create_least_core_problem(u: UtilityBase, sampler: IndexSampler, n_iterations: int):
    n_obs = len(u.training_data)

    A_lb = np.zeros((n_iterations, n_obs))
    utility_values = np.zeros(n_iterations)

    generator = sampler.from_indices(u.training_data.indices)
    for i, batch in enumerate(islice(generator, n_iterations)):
        sample = list(batch)[0]
        A_lb[i, sample.subset.astype(int)] = 1
        utility_values[i] = u(sample)

    return LeastCoreProblem(utility_values=utility_values, A_lb=A_lb)

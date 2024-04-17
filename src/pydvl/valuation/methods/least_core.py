from __future__ import annotations

from enum import Enum

from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods._montecarlo_least_core import montecarlo_least_core
from pydvl.valuation.methods._naive_least_core import exact_least_core
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
        n_jobs: int = 1,
        n_iterations: int | None = None,
        mode: LeastCoreMode = LeastCoreMode.MonteCarlo,
        non_negative_subsidy: bool = False,
        solver_options: dict | None = None,
        progress: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.utility = utility
        self.n_jobs = n_jobs
        self.n_iterations = n_iterations
        # self.sampler = sampler
        self.mode = mode
        self.non_negative_subsidy = non_negative_subsidy
        self.solver_options = solver_options
        self.progress = progress
        self.kwargs = kwargs

    def fit(self, data: Dataset) -> None:

        self.utility = self.utility.with_dataset(data)

        if self.mode == LeastCoreMode.MonteCarlo:
            # TODO fix progress showing in remote case
            self.progress = False
            if self.n_iterations is None:
                raise ValueError(
                    "n_iterations cannot be None for Monte Carlo Least Core"
                )
            values = montecarlo_least_core(  # type: ignore
                u=self.utility,
                n_iterations=self.n_iterations,
                n_jobs=self.n_jobs,
                progress=self.progress,
                non_negative_subsidy=self.non_negative_subsidy,
                solver_options=self.solver_options,
                **self.kwargs,
            )
        elif self.mode == LeastCoreMode.Exact:
            values = exact_least_core(
                u=self.utility,
                progress=self.progress,
                non_negative_subsidy=self.non_negative_subsidy,
                solver_options=self.solver_options,
            )

        else:
            raise ValueError(f"Invalid value encountered in {mode=}")

        self.result = values

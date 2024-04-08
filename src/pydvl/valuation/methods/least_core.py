from __future__ import annotations

from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.samplers.powerset import PowersetSampler
from pydvl.valuation.utility.base import UtilityBase

__all__ = ["LeastCoreValuation"]


class LeastCoreValuation(Valuation):
    def __init__(
        self,
        utility: UtilityBase,
        sampler: PowersetSampler,
        n_constraints: int,
        non_negative_subsidy: float,
        solver_options: dict | None = None,
    ):
        super().__init__()
        self.utility = utility
        self.sampler = sampler
        self.n_constraints = n_constraints
        self.non_negative_subsidy = non_negative_subsidy
        self.solver_options = solver_options

    def fit(self, data: Dataset) -> None:
        pass

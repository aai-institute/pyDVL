import math

from pydvl.valuation.methods.semivalue import SemivalueValuation

__all__ = ["DataShapleyValuation"]


class DataShapleyValuation(SemivalueValuation):
    """Computes Shapley values."""

    algorithm_name = "Data-Shapley"

    def coefficient(self, n: int, k: int, weight: float) -> float:
        return weight / math.comb(n - 1, k) / n

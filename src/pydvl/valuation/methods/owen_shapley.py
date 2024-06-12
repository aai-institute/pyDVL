from __future__ import annotations

from pydvl.utils import Status
from pydvl.valuation.methods.semivalue import SemivalueValuation
from pydvl.valuation.samplers.powerset import OwenSampler
from pydvl.valuation.stopping import NoStopping
from pydvl.valuation.utility.base import UtilityBase

"""
## References

[^1]: <a name="okhrati_multilinear_2021"></a>Okhrati, R., Lipani, A., 2021.
    [A Multilinear Sampling Algorithm to Estimate Shapley Values](https://ieeexplore.ieee.org/abstract/document/9412511).
    In: 2020 25th International Conference on Pattern Recognition (ICPR), pp. 7992–7999. IEEE.
"""


class OwenShapleyValuation(SemivalueValuation):
    def __init__(
        self,
        utility: UtilityBase,
        sampler: OwenSampler,
        progress: dict[str, Any] | bool = False,
    ):
        super().__init__(
            utility=utility,
            sampler=sampler,
            is_done=NoStopping(),
            progress=progress,
        )

    def fit(self, dataset: Dataset) -> ValuationResult:
        # since we bypassed the convergence checks we need to set the status to
        # converged manually
        super().fit(dataset)
        self.result._status = Status.Converged
        return self

    def coefficient(self, n: int, k: int) -> float:
        return 1

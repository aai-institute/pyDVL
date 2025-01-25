from __future__ import annotations

from typing import Any

from typing_extensions import Self

from pydvl.utils import Status
from pydvl.valuation.dataset import Dataset
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
    """Umbrella class to calculate Shapley values with Owen sampling schemes.

    Owen shapley values converge to true Shapley values as the number of samples
    increases but have been shown to need fewer samples than other sampling schemes.

    The number of samples is governed by the sampler object. There are no convergence
    criteria for Owen shapley values as they will just run for a fixed number of
    samples.

    Args:
        utility: Utility object with model and scoring function.
        sampler: Owen sampling scheme to use. Can be OwenSampler or
            AntitheticOwenSampler.
        progress: Whether to show a progress bar.

    """

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

    def fit(self, dataset: Dataset) -> Self:
        """Calculate the Owen shapley values for a given dataset.

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
        # since we bypassed the convergence checks we need to set the status to
        # converged manually
        super().fit(dataset)
        # make the type checker happy
        if self.result is not None:
            self.result._status = Status.Converged
        return self

    def coefficient(self, n: int, k: int, weight: float) -> float:
        # Coefficient is 1.0 for all n and k
        return weight

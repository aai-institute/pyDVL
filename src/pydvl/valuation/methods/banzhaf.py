r"""
This module implements the Banzhaf valuation method, as described in
Wang and Jia, (2022)[^1].

Data Banzhaf was proposed as a means to counteract the inherent stochasticity of the
utility function in machine learning problems. It chooses the coefficients $w(k)$ of the
semi-value valuation function to be constant $2^{n-1}$ for all set sizes $k,$ yielding:

$$
v_\text{bzf}(i) = \frac{1}{2^{n-1}} \sum_{S \sim P(D_{-i})} [u(S_{+i}) - u(S)],
$$

!!! info "Background on semi-values"
    The Banzhaf valuation is a special case of the semi-value valuation method. You can
    read a short introduction [in the documentation][semi-values-intro].

The intuition for picking a constant weight is that for any choice of weight function
$w$, one can always construct a utility with higher variance where $w$ is greater.
Therefore, in a worst-case sense, the best one can do is to pick a constant weight.

Data Banzhaf proves to outperform many other valuation methods in downstream tasks like
best point removal.


## Maximum Sample Reuse Banzhaf

A special sampling scheme (MSR) that reuses each sample to update every index in the
dataset is shown by Wang and Jia to be optimal for the Banzhaf valuation: not only does
it drastically reduce the number of sets needed, but the sampling distribution also
matches the Banzhaf indices, in the sense explained in [Sampling strategies for
semi-values][semi-values-sampling].

In order to work with this sampler for Banzhaf values, you can use
[MSRBanzhafValuation][pydvl.valuation.methods.banzhaf.MSRBanzhafValuation]. In
principle, it is also possible to select the
[MSRSampler][pydvl.valuation.samplers.msr.MSRSampler] when instantiating
[BanzhafValuation][pydvl.valuation.methods.banzhaf.BanzhafValuation], but this might
introduce some numerical instability, as explained in the document linked above.


## References

[^1]: <a name="wang_data_2023"></a> Wang, Jiachen T., and Ruoxi Jia. [Data Banzhaf: A
      Robust Data Valuation Framework for Machine
      Learning](https://proceedings.mlr.press/v206/wang23e.html). In Proceedings of The
      26th International Conference on Artificial Intelligence and Statistics,
      6388–6421. PMLR, 2023.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pydvl.utils.types import Seed
from pydvl.valuation.methods.semivalue import SemivalueValuation
from pydvl.valuation.samplers.msr import MSRSampler
from pydvl.valuation.stopping import StoppingCriterion
from pydvl.valuation.types import SemivalueCoefficient

__all__ = ["BanzhafValuation", "MSRBanzhafValuation"]

from pydvl.valuation.utility.base import UtilityBase


class BanzhafValuation(SemivalueValuation):
    """Computes Banzhaf values."""

    algorithm_name = "Data-Banzhaf"

    @property
    def log_coefficient(self) -> SemivalueCoefficient | None:
        """Returns the log-coefficient of the Banzhaf valuation."""

        def _log_coefficient(n: int, k: int) -> float:
            return float(-(n - 1) * np.log(2) - self.sampler.log_weight(n, k))

        return _log_coefficient


class MSRBanzhafValuation(SemivalueValuation):
    """Computes Banzhaf values with Maximum Sample Reuse.

    This can be seen as a convenience class that wraps the
    [MSRSampler][pydvl.valuation.samplers.msr.MSRSampler] but in fact it also skips
    importance sampling altogether, since the MSR sampling scheme already provides the
    correct weights for the Monte Carlo approximation. This can avoid some numerical
    inaccuracies that can arise, when using an `MSRSampler` with
    [BanzhafValuation][pydvl.valuation.methods.banzhaf.BanzhafValuation], despite the
    fact that the respective coefficients cancel each other out analytically.
    """

    algorithm_name = "Data-Banzhaf-MSR"

    def __init__(
        self,
        utility: UtilityBase,
        is_done: StoppingCriterion,
        batch_size: int = 1,
        seed: Seed | None = None,
        skip_converged: bool = False,
        show_warnings: bool = True,
        progress: dict[str, Any] | bool = False,
    ):
        sampler = MSRSampler(batch_size=batch_size, seed=seed)
        super().__init__(
            utility, sampler, is_done, skip_converged, show_warnings, progress
        )

    @property
    def log_coefficient(self) -> SemivalueCoefficient | None:
        """Disable importance sampling for this method since we have a fixed sampler
        that already provides the correct weights for the Monte Carlo approximation."""
        return None

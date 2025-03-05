r"""
This module implements the Banzhaf valuation method, as described in
Wang and Jia, (2022)<sup><a href="#wang_data_2023">1</a></sup>.

Data Banzhaf was proposed as a means to counteract the inherent stochasticity of the
utility function in machine learning problems. It chooses the coefficients $w(k)$ of the
semi-value valuation function to be constant:

$$w(k) := 2^{n-1},$$

for all set sizes $k$. The intuition for picking a constant weight is that for
any choice of weight function $w$, one can always construct a utility with
higher variance where $w$ is greater. Therefore, in a worst-case sense, the best
one can do is to pick a constant weight.

Data Banzhaf proves to outperform many other valuation methods in downstream tasks like
best point removal, but can show some

## References

[^1]: <a name="wang_data_2023"></a> Wang, Jiachen T., and Ruoxi Jia. [Data Banzhaf: A
      Robust Data Valuation Framework for Machine
      Learning](https://proceedings.mlr.press/v206/wang23e.html). In Proceedings of The
      26th International Conference on Artificial Intelligence and Statistics,
      6388–6421. PMLR, 2023.
"""

import numpy as np

from pydvl.valuation.methods.semivalue import SemivalueValuation

__all__ = ["BanzhafValuation"]


class BanzhafValuation(SemivalueValuation):
    """Computes Banzhaf values."""

    algorithm_name = "Data-Banzhaf"

    def log_coefficient(self, n: int, k: int) -> float:
        return float(-(n - 1) * np.log(2))

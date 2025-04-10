"""
This module will implement 2D-Shapley, as introduced in (Liu et al., 2023)<sup><a
href="#liu_2dshapley_2023">1</a></sup>.

!!! failure "To do"
    Once this is implemented, remember to un-exclude the file in
    /build_scripts/generate_api_docs.py.

## References

[^1]: <a name="liu_2dshapley_2023"></a>Liu, Zhihong, Hoang Anh Just, Xiangyu Chang, Xi
      Chen, and Ruoxi Jia. [2D-Shapley: A Framework for Fragmented Data
      Valuation](https://proceedings.mlr.press/v202/liu23s.html). In Proceedings of the
      40th International Conference on Machine Learning, 21730–55. PMLR, 2023.

"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from numpy.typing import NDArray
from typing_extensions import Self

from pydvl.valuation.base import Valuation
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.types import IndexT, Sample


@dataclass(frozen=True)
class TwoDSample(Sample):
    """A sample for 2D-Shapley, consisting of a set of indices and a set of features."""

    features: NDArray[IndexT]

    def __hash__(self):
        array_bytes = self.subset.tobytes() + self.features.tobytes()
        sha256_hash = hashlib.sha256(array_bytes).hexdigest()
        return int(sha256_hash, base=16)


class TwoDShapley(Valuation):
    algorithm_name: str = "2D-Shapley"

    def fit(self, data: Dataset, continue_from: ValuationResult | None = None) -> Self:
        # With the right sampler and a subclassed utility, this should follow a very
        # similar pattern to the other methods.
        # Note that it should be trivial to generalize to other coefficients, sampling
        # strategies, etc.
        raise NotImplementedError

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import cast

from numpy.typing import NDArray

from pydvl.valuation.scorers.classwise import ClasswiseSupervisedScorer
from pydvl.valuation.types import IndexT, Sample
from pydvl.valuation.utility import Utility

__all__ = ["CSSample", "ClasswiseUtility"]


@dataclass(frozen=True)
class CSSample(Sample):
    label: int | None
    in_class_subset: NDArray[IndexT]

    # Make the unpacking operator work
    def __iter__(self):  # No way to type the return Iterator properly
        return iter((self.idx, self.subset, self.label, self.in_class_subset))

    def __hash__(self):
        array_bytes = self.subset.tobytes() + self.in_class_subset.tobytes()
        sha256_hash = hashlib.sha256(array_bytes).hexdigest()
        return int(sha256_hash, base=16)


class ClasswiseUtility(Utility[CSSample]):
    """
    FIXME: probably unnecessary, just a test
    """

    scorer: ClasswiseSupervisedScorer

    def __call__(self, sample: CSSample) -> float:
        return cast(float, self._utility_wrapper(sample))

    def _utility(self, sample: CSSample) -> float:
        self.scorer.label = sample.label
        # TODO: do the thing
        raise NotImplementedError

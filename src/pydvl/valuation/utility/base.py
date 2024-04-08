from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic

import numpy as np
from numpy.typing import NDArray

from pydvl.utils import SupervisedModel
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.scorers.scorer import Scorer
from pydvl.valuation.types import SampleT


class UtilityBase(Generic[SampleT], ABC):
    model: SupervisedModel
    test_data: Dataset
    training_data: Dataset | None
    scorer: Scorer
    default_score: float
    score_range: NDArray[np.float_]

    @abstractmethod
    def __call__(self, sample: SampleT) -> float:
        ...

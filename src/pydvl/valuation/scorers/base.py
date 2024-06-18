"""
Scorers are a fundamental building block of many data valuation methods. They
are typically used by [Utility][pydvl.valuation.utility.Utility] and its subclasses
to evaluate the quality of a model when trained on subsets of the training data.

Scorers evaluate trained models in user-defined ways, and provide additional
information about themselves, like their range and default value, which can be used by
some data valuation methods (e.g. [Group Testing
Shapley][pydvl.valuation.methods.gt_shapley]) to estimate the number of samples required
for a certain quality of approximation.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Scorer(ABC):
    """
    A scoring callable that takes a model and returns a scalar.

    !!! tip "Added in version 0.10.0"
        ABC added
    """

    default: float
    name: str
    range: NDArray[np.float_]

    @abstractmethod
    def __call__(self, model) -> float:
        ...

from enum import Enum
from typing import Protocol

import numpy as np
from numpy._typing import NDArray


class ShapleyMode(str, Enum):
    """Supported algorithms for the computation of Shapley values.

    .. todo::
       Make algorithms register themselves here.
    """

    CombinatorialExact = "combinatorial_exact"
    CombinatorialMontecarlo = "combinatorial_montecarlo"
    GroupTesting = "group_testing"
    KNN = "knn"
    Owen = "owen"
    OwenAntithetic = "owen_antithetic"
    PermutationExact = "permutation_exact"
    PermutationMontecarlo = "permutation_montecarlo"
    TruncatedMontecarlo = "truncated_montecarlo"


class PermutationBreaker(Protocol):
    def __call__(self, idx: int, marginals: NDArray[np.float_]) -> bool:
        ...


class ValueStopper(Protocol):
    def __call__(
        self, marginals: NDArray[np.float_], variances: NDArray[np.float_]
    ) -> bool:
        ...

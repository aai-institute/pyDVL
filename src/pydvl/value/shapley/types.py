from enum import Enum


class ShapleyMode(str, Enum):
    """Supported algorithms for the computation of Shapley values.

    !!! Todo
       Make algorithms register themselves here.
    """

    ApproShapley = "appro_shapley"  # Alias for PermutationMontecarlo
    CombinatorialExact = "combinatorial_exact"
    CombinatorialMontecarlo = "combinatorial_montecarlo"
    GroupTesting = "group_testing"
    KNN = "knn"
    Owen = "owen"
    OwenAntithetic = "owen_antithetic"
    PermutationExact = "permutation_exact"
    PermutationMontecarlo = "permutation_montecarlo"
    TruncatedMontecarlo = "truncated_montecarlo"

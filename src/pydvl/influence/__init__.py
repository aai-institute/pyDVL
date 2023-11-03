"""
This package contains algorithms for the computation of the influence function.

> **Warning:** Much of the code in this package is experimental or untested and is subject to modification.
In particular, the package structure and basic API will probably change.

"""
from .general import (
    InfluenceType,
    compute_influence_factors,
    compute_influences,
    compute_influences_pert,
    compute_influences_up,
)
from .inversion import InversionMethod

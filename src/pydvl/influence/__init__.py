"""
This package contains algorithms for the computation of the influence function.

See [The Influence function][influence-function] for an introduction to the
concepts and methods implemented here.

!!! Warning
    Much of the code in this package is experimental or untested and is subject
    to modification. In particular, the package structure and basic API will
    probably change.

"""

from .influence_calculator import (  # noqa: F401
    DaskInfluenceCalculator,
    DisableClientSingleThreadCheck,
    SequentialInfluenceCalculator,
)
from .types import InfluenceMode  # noqa: F401

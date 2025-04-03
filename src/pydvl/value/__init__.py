"""
This module implements algorithms for the exact and approximate computation of
values and semi-values.

See [Data valuation][data-valuation-intro] for an introduction to the concepts
and methods implemented here.
"""

import warnings

msg = (
    "The package pydvl.value was deprecated since v0.10.0 in favor of "
    "pydvl.valuation. It will be removed in v0.12.0."
)

warnings.warn(msg, FutureWarning)

from ..utils import Dataset, Scorer, Utility  # noqa
from .least_core import *  # noqa: E402
from .loo import *  # noqa: E402
from .oob import *  # noqa: E402
from .result import *  # noqa: E402
from .sampler import *  # noqa: E402
from .semivalues import *  # noqa: E402
from .shapley import *  # noqa: E402
from .stopping import *  # noqa: E402

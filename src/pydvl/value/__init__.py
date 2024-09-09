"""
This module implements algorithms for the exact and approximate computation of
values and semi-values.

See [Data valuation][data-valuation] for an introduction to the concepts
and methods implemented here.
"""

import warnings

msg = (
    "The package pydvl.value was deprecated since v0.10.0 in favor of "
    "pydvl.valuation. It will be removed in v0.12.0."
)

warnings.warn(msg, FutureWarning)

from .result import *  # isort: skip
from ..utils import Dataset, Scorer, Utility
from .least_core import *
from .loo import *
from .oob import *
from .sampler import *
from .semivalues import *
from .shapley import *
from .stopping import *

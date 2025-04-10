"""
This module contains mixin classes.

Currently only one for samplers which use a random number generator.
"""

from __future__ import annotations

import numpy as np

from pydvl.utils.types import Seed
from pydvl.valuation.types import IndexSetT


class StochasticSamplerMixin:
    """Mixin class for samplers which use a random number generator.
    Args:
        seed: Seed for the random number generator. Passed to
            [numpy.random.default_rng][].
    """

    def __init__(self, *args, seed: Seed | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng(seed)

    def sample_limit(self, indices: IndexSetT) -> int | None:
        return None

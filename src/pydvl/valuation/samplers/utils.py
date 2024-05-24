from __future__ import annotations

import numpy as np

from pydvl.utils.types import Seed


class StochasticSamplerMixin:
    """Mixin class for samplers which use a random number generator."""

    def __init__(self, *args, seed: Seed | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng(seed)

import numpy as np
import pytest

from pydvl.valuation import (
    AntitheticOwenSampler,
    GridOwenStrategy,
    NoStopping,
    OwenSampler,
    ShapleyValuation,
    UniformOwenStrategy,
)
from tests.valuation import recursive_make


# @pytest.mark.skip(reason="An unnecessary test of numerical stability")
@pytest.mark.parametrize("sampler_cls", [OwenSampler, AntitheticOwenSampler])
@pytest.mark.parametrize(
    "sampler_kwargs",
    [
        {"outer_sampling_strategy": (GridOwenStrategy, {"n_samples_outer": 10})},
        {"outer_sampling_strategy": (UniformOwenStrategy, {"n_samples_outer": 10})},
    ],
)
def test_owen_weight(sampler_cls, sampler_kwargs, dummy_utility):
    """This tests that we effectively cancel the Shapley coefficient using the Owen
    samplers, so that the method has a coefficient of 1.0 for all combinations of n and
    k.

    It is actually not necessary to test this, but I'm leaving the code here in case we
    want to come back to it later.
    """
    sampler = recursive_make(sampler_cls, sampler_kwargs)
    utility = dummy_utility
    valuation = ShapleyValuation(utility, sampler, is_done=NoStopping())

    results = []
    coeff = valuation.log_coefficient
    for n in np.random.randint(1000, 100000, size=4):  # type: int
        for k in np.random.randint(0, n, size=4):  # type: int
            results.append(coeff(n, k))  # noqa: F821

    np.testing.assert_allclose(np.exp(results), 1.0)

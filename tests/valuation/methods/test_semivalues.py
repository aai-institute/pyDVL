import math
from typing import Type

import numpy as np
import pytest
from joblib import parallel_config

from pydvl.utils.types import Seed
from pydvl.valuation import (
    BetaShapleyValuation,
    DataBanzhafValuation,
    DataShapleyValuation,
    DeltaShapleyValuation,
    MSRBanzhafValuation,
    SemivalueValuation,
)
from pydvl.valuation.samplers import (
    AntitheticPermutationSampler,
    AntitheticSampler,
    DeterministicPermutationSampler,
    DeterministicUniformSampler,
    MSRSampler,
    PermutationSampler,
    PowersetSampler,
    UniformSampler,
)
from pydvl.valuation.samplers.utils import StochasticSamplerMixin
from pydvl.value.stopping import HistoryDeviation, MaxChecks, MaxUpdates, MinUpdates

from .. import check_values
from ..utils import timed


@pytest.mark.flaky(reruns=1)
@pytest.mark.parametrize("num_samples", [5])
def test_msr_banzhaf(
    num_samples: int,
    analytic_banzhaf,
    dummy_train_data,
    n_jobs,
    seed: Seed,
):
    u, exact_values = analytic_banzhaf

    valuation = MSRBanzhafValuation(
        utility=u,
        sampler=MSRSampler(seed=seed),
        is_done=MinUpdates(500 * num_samples),
        progress=False,
    )
    with parallel_config(n_jobs=n_jobs):
        valuation.fit(dummy_train_data)

    values = valuation.values()

    check_values(values, exact_values, atol=0.025)

    # Check order
    assert np.array_equal(np.argsort(exact_values), np.argsort(values))


@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize(
    "valuation_class, kwargs",
    [
        (BetaShapleyValuation, {"alpha": 1, "beta": 1}),
        (BetaShapleyValuation, {"alpha": 1, "beta": 16}),
        (BetaShapleyValuation, {"alpha": 4, "beta": 1}),
        (DataBanzhafValuation, {}),
        (DataShapleyValuation, {}),
    ],
)
def test_coefficients(n, valuation_class, kwargs):
    r"""Coefficients for semi-values must fulfill:

    $$ \sum_{i=1}^{n}\choose{n-1}{j-1}w^{(n)}(j) = 1 $$

    Note that we depart from the usual definitions by including the factor $1/n$
    in the shapley and beta coefficients.
    """
    valuation = valuation_class(
        utility=None,
        sampler=UniformSampler(),
        is_done=MaxUpdates(50),
        progress=False,
        **kwargs,
    )

    s = [
        math.comb(n - 1, j - 1) * valuation.coefficient(n, j - 1)
        for j in range(1, n + 1)
    ]
    assert np.isclose(1, np.sum(s))


@pytest.mark.parametrize(
    "test_game",
    [
        ("shoes", {"left": 3, "right": 2}),
    ],
    indirect=["test_game"],
)
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_shapley_batch_size(
    test_game,
    n_jobs,
    seed,
):
    def compute_semivalues(batch_size, n_jobs=n_jobs, seed=seed):
        valuation = BetaShapleyValuation(
            utility=test_game.u,
            sampler=UniformSampler(batch_size=batch_size, seed=seed),
            is_done=MaxUpdates(100),
            progress=False,
            alpha=1,
            beta=1,
        )
        with parallel_config(n_jobs=n_jobs):
            valuation.fit(test_game.data)
        return valuation.values()

    timed_fn = timed(compute_semivalues)
    result_single_batch = timed_fn(batch_size=1)
    result_multi_batch = timed_fn(batch_size=5)

    # Occasionally, batch_2 arrives before batch_1, so rtol isn't always 0.
    check_values(result_single_batch, result_multi_batch, rtol=1e-4)


@pytest.mark.parametrize("num_samples", [5])
@pytest.mark.parametrize(
    "sampler_class",
    [
        DeterministicUniformSampler,
        DeterministicPermutationSampler,
        UniformSampler,
        PermutationSampler,
        AntitheticSampler,
        AntitheticPermutationSampler,
    ],
)
def test_banzhaf(
    num_samples: int,
    analytic_banzhaf,
    dummy_train_data,
    sampler_class: Type[PowersetSampler],
    n_jobs: int,
    seed,
):
    u, exact_values = analytic_banzhaf

    if issubclass(sampler_class, StochasticSamplerMixin):
        sampler = sampler_class(seed=seed)
    else:
        sampler = sampler_class()

    valuation = DataBanzhafValuation(
        utility=u,
        sampler=sampler,
        is_done=HistoryDeviation(50, 1e-3) | MaxUpdates(1000),
        progress=False,
    )

    with parallel_config(n_jobs=n_jobs):
        valuation.fit(dummy_train_data)
    values = valuation.values()

    check_values(values, exact_values, rtol=0.2)

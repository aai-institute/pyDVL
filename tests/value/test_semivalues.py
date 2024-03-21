import math
from itertools import islice
from typing import Type

import numpy as np
import pytest

from pydvl.parallel.config import ParallelConfig
from pydvl.utils.types import Seed
from pydvl.value.sampler import (
    AntitheticPermutationSampler,
    AntitheticSampler,
    DeterministicPermutationSampler,
    DeterministicUniformSampler,
    PermutationSampler,
    PowersetSampler,
    UniformSampler,
)
from pydvl.value.semivalues import (
    MSRMarginal,
    SVCoefficient,
    _marginal,
    banzhaf_coefficient,
    beta_coefficient,
    compute_generic_semivalues,
    msr_banzhaf,
    shapley_coefficient,
)
from pydvl.value.stopping import HistoryDeviation, MaxUpdates, RankStability

from . import check_values
from .utils import timed


@pytest.mark.parametrize(
    "test_game",
    [
        ("shoes", {"left": 3, "right": 2}),
    ],
    indirect=["test_game"],
)
@pytest.mark.parametrize(
    "sampler, coefficient, batch_size",
    [(PermutationSampler, beta_coefficient(1, 1), 5)],
)
def test_marginal_batch_size(test_game, sampler, coefficient, batch_size, seed):
    # TODO: This test is probably not needed.
    # Because I added it and then realized that it doesn't do much.
    # The only difference between the two calls is that for the first one
    # the loop is outside and the second one the loop is inside.
    sampler_it = iter(sampler(test_game.u.data.indices, seed=seed))
    samples = tuple(islice(sampler_it, batch_size))

    marginals_single = []
    for sample in samples:
        marginals_single.extend(
            _marginal(test_game.u, coefficient=coefficient, samples=[sample])
        )

    marginals_batch = _marginal(test_game.u, coefficient=coefficient, samples=samples)

    assert len(marginals_single) == len(marginals_batch)
    assert set(marginals_single) == set(marginals_batch)


@pytest.mark.parametrize("num_samples", [5])
def test_msr_banzhaf(
    num_samples: int, analytic_banzhaf, parallel_config, n_jobs, seed: Seed
):
    u, exact_values = analytic_banzhaf
    sampler = MSRSampler()
    marginal = MSRMarginal()
    values = compute_generic_semivalues(
        sampler(u.data.indices, seed=seed),
        u=u,
        coefficient=coefficient,
        marginal=marginal,
        criterion=RankStability(rtol=0.1) | MaxUpdates(100),
        skip_converged=False,
        n_jobs=n_jobs,
        config=parallel_config,
        progress=True,
    )
    check_values(values, exact_values, rtol=0.1)


@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize(
    "coefficient",
    [
        beta_coefficient(1, 1),
        beta_coefficient(1, 16),
        beta_coefficient(4, 1),
        banzhaf_coefficient,
        shapley_coefficient,
    ],
)
def test_coefficients(n: int, coefficient: SVCoefficient):
    r"""Coefficients for semi-values must fulfill:

    $$ \sum_{i=1}^{n}\choose{n-1}{j-1}w^{(n)}(j) = 1 $$

    Note that we depart from the usual definitions by including the factor $1/n$
    in the shapley and beta coefficients.
    """
    s = [math.comb(n - 1, j - 1) * coefficient(n, j - 1) for j in range(1, n + 1)]
    assert np.isclose(1, np.sum(s))


@pytest.mark.parametrize(
    "test_game",
    [
        ("symmetric-voting", {"n_players": 4}),
        ("shoes", {"left": 1, "right": 1}),
        ("shoes", {"left": 2, "right": 1}),
        ("shoes", {"left": 1, "right": 2}),
    ],
    indirect=["test_game"],
)
@pytest.mark.parametrize(
    "sampler",
    [
        DeterministicUniformSampler,
        DeterministicPermutationSampler,
    ],
)
@pytest.mark.parametrize("coefficient", [shapley_coefficient, beta_coefficient(1, 1)])
def test_games_shapley_deterministic(
    test_game,
    parallel_config,
    n_jobs,
    sampler: Type[PowersetSampler],
    coefficient: SVCoefficient,
    seed: Seed,
):
    criterion = MaxUpdates(50)
    values = compute_generic_semivalues(
        sampler(test_game.u.data.indices, seed=seed),
        test_game.u,
        coefficient,
        criterion,
        skip_converged=True,
        n_jobs=n_jobs,
        config=parallel_config,
        progress=True,
    )
    exact_values = test_game.shapley_values()
    check_values(values, exact_values, rtol=0.1)


@pytest.mark.parametrize(
    "test_game",
    [
        ("symmetric-voting", {"n_players": 6}),
        ("shoes", {"left": 3, "right": 2}),
    ],
    indirect=["test_game"],
)
@pytest.mark.parametrize(
    "sampler",
    [
        UniformSampler,
        PermutationSampler,
        pytest.param(AntitheticSampler, marks=pytest.mark.slow),
        AntitheticPermutationSampler,
    ],
)
@pytest.mark.parametrize("coefficient", [shapley_coefficient, beta_coefficient(1, 1)])
def test_games_shapley(
    test_game,
    parallel_config,
    n_jobs,
    sampler: Type[PowersetSampler],
    coefficient: SVCoefficient,
    seed: Seed,
):
    criterion = HistoryDeviation(50, 1e-4) | MaxUpdates(500)
    values = compute_generic_semivalues(
        sampler(test_game.u.data.indices, seed=seed),
        test_game.u,
        coefficient,
        criterion,
        skip_converged=True,
        n_jobs=n_jobs,
        config=parallel_config,
        progress=True,
    )

    exact_values = test_game.shapley_values()
    check_values(values, exact_values, rtol=0.2)


@pytest.mark.parametrize(
    "test_game",
    [
        ("shoes", {"left": 3, "right": 2}),
    ],
    indirect=["test_game"],
)
@pytest.mark.parametrize(
    "sampler, coefficient, batch_size",
    [(PermutationSampler, beta_coefficient(1, 1), 5)],
)
@pytest.mark.parametrize(
    "n_jobs",
    [
        1,
        pytest.param(
            2,
            marks=pytest.mark.xfail(
                reason="Bad interaction between parallelization and batching"
            ),
        ),
    ],
)
def test_shapley_batch_size(
    test_game,
    sampler: Type[PermutationSampler],
    coefficient: SVCoefficient,
    batch_size: int,
    n_jobs: int,
    parallel_config: ParallelConfig,
    seed: Seed,
):
    timed_fn = timed(compute_generic_semivalues)
    result_single_batch = timed_fn(
        sampler(test_game.u.data.indices, seed=seed),
        test_game.u,
        coefficient,
        done=MaxUpdates(100),
        skip_converged=True,
        n_jobs=n_jobs,
        batch_size=1,
        config=parallel_config,
    )
    total_seconds_single_batch = timed_fn.execution_time

    result_multi_batch = timed_fn(
        sampler(test_game.u.data.indices, seed=seed),
        test_game.u,
        coefficient,
        done=MaxUpdates(100),
        skip_converged=True,
        n_jobs=n_jobs,
        batch_size=batch_size,
        config=parallel_config,
    )
    total_seconds_multi_batch = timed_fn.execution_time
    assert total_seconds_multi_batch < total_seconds_single_batch * 1.1

    # Occasionally, batch_2 arrives before batch_1, so rtol isn't always 0.
    check_values(result_single_batch, result_multi_batch, rtol=1e-4)


@pytest.mark.parametrize("num_samples", [5])
@pytest.mark.parametrize(
    "sampler",
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
    sampler: Type[PowersetSampler],
    n_jobs: int,
    parallel_config: ParallelConfig,
    seed,
):
    u, exact_values = analytic_banzhaf
    criterion = HistoryDeviation(50, 1e-3) | MaxUpdates(1000)
    values = compute_generic_semivalues(
        sampler(u.data.indices, seed=seed),
        u,
        banzhaf_coefficient,
        criterion,
        skip_converged=True,
        n_jobs=n_jobs,
        config=parallel_config,
    )
    check_values(values, exact_values, rtol=0.2)

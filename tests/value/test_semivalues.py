import math
from itertools import islice
from typing import Type

import numpy as np
import pytest

from pydvl.parallel.config import ParallelConfig
from pydvl.utils.functional import timed
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
    DefaultMarginal,
    MSRFutureProcessor,
    SVCoefficient,
    banzhaf_coefficient,
    beta_coefficient,
    compute_generic_semivalues,
    compute_msr_banzhaf_semivalues,
    shapley_coefficient,
)
from pydvl.value.stopping import HistoryDeviation, MaxChecks, MaxUpdates

from . import check_values


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
            DefaultMarginal()(test_game.u, coefficient=coefficient, samples=[sample])
        )

    marginals_batch = DefaultMarginal()(
        test_game.u, coefficient=coefficient, samples=samples
    )

    assert len(marginals_single) == len(marginals_batch)
    assert set(marginals_single) == set(marginals_batch)


@pytest.mark.parametrize("num_samples", [5])
def test_msr_banzhaf(
    num_samples: int, analytic_banzhaf, parallel_backend, n_jobs, seed: Seed
):
    u, exact_values = analytic_banzhaf
    values = compute_msr_banzhaf_semivalues(
        u=u,
        done=MaxChecks(200 * num_samples),
        parallel_backend=parallel_backend,
        n_jobs=n_jobs,
        seed=seed,
    )
    # Need to use atol because msr banzhaf is quite noisy.
    check_values(values, exact_values, atol=0.1)

    # Check order
    assert np.array_equal(np.argsort(exact_values), np.argsort(values))


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
    np.testing.assert_allclose(1, np.sum(s))


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
    parallel_backend,
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
        parallel_backend=parallel_backend,
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
    parallel_backend,
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
        parallel_backend=parallel_backend,
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
    parallel_backend: ParallelConfig,
    seed: Seed,
):
    timed_fn = timed()(compute_generic_semivalues)
    result_single_batch = timed_fn(
        sampler(test_game.u.data.indices, seed=seed),
        test_game.u,
        coefficient,
        done=MaxUpdates(100),
        skip_converged=True,
        n_jobs=n_jobs,
        batch_size=1,
        parallel_backend=parallel_backend,
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
        parallel_backend=parallel_backend,
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
    parallel_backend: ParallelConfig,
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
        parallel_backend=parallel_backend,
    )
    check_values(values, exact_values, rtol=0.2)


@pytest.mark.parametrize("num_samples", [6])
def test_msr_future_processor(num_samples: int, dummy_utility):
    proc = MSRFutureProcessor(dummy_utility)
    assert proc.n == num_samples
    assert proc.total_evaluations == 0
    data1 = [1, 3, 5]
    data2 = [0, 2, 4]

    # Iteration 1
    marginals1 = proc([(data1, 1.0)])
    assert marginals1 == [[(i, 0.0) for i in range(6)]]

    # Iteration 2
    marginals2 = proc([(data2, 0.5)])
    assert marginals2 == [
        [(0, -1.0), (1, 1.0), (2, -1.0), (3, 1.0), (4, -1.0), (5, 1.0)]
    ]

    # Iteration 3
    # Values are -0.5 for even and 0.5 for uneven indices
    # New values are supposed to be -1.0 and 1.0=(1.0+2.0)/2-(0.5)/1
    # Marginals need to be 2 and -2 because (2*0.5 + 1*2)/3=1
    marginals2 = proc([(data1, 2.0)])
    assert marginals2 == [
        [(0, -2.0), (1, 2.0), (2, -2.0), (3, 2.0), (4, -2.0), (5, 2.0)]
    ]

import math
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
    SVCoefficient,
    banzhaf_coefficient,
    beta_coefficient,
    compute_generic_semivalues,
    shapley_coefficient,
)
from pydvl.value.stopping import HistoryDeviation, MaxUpdates

from . import check_values
from .utils import timed


@pytest.mark.parametrize("num_samples", [5])
@pytest.mark.parametrize(
    "sampler",
    [
        DeterministicUniformSampler,
        DeterministicPermutationSampler,
        UniformSampler,
        PermutationSampler,
        pytest.param(AntitheticSampler, marks=pytest.mark.slow),
        AntitheticPermutationSampler,
    ],
)
@pytest.mark.parametrize("coefficient", [shapley_coefficient, beta_coefficient(1, 1)])
def test_shapley(
    num_samples: int,
    analytic_shapley,
    sampler: Type[PowersetSampler],
    coefficient: SVCoefficient,
    n_jobs: int,
    parallel_config: ParallelConfig,
):
    u, exact_values = analytic_shapley
    criterion = HistoryDeviation(50, 1e-3) | MaxUpdates(1000)
    values = compute_generic_semivalues(
        sampler(u.data.indices),
        u,
        coefficient,
        criterion,
        skip_converged=True,
        n_jobs=n_jobs,
        config=parallel_config,
    )
    check_values(values, exact_values, rtol=0.2)


@pytest.mark.parametrize(
    "num_samples,sampler,coefficient,batch_size",
    [(5, PermutationSampler, beta_coefficient(1, 1), 5)],
)
def test_shapley_batch_size(
    num_samples: int,
    analytic_shapley,
    sampler: Type[PermutationSampler],
    coefficient: SVCoefficient,
    batch_size: int,
    n_jobs: int,
    parallel_config: ParallelConfig,
    seed: Seed,
):
    u, exact_values = analytic_shapley
    timed_fn = timed(compute_generic_semivalues)
    result_single_batch = timed_fn(
        sampler(u.data.indices, seed=seed),
        u,
        coefficient,
        done=HistoryDeviation(50, 1e-3) | MaxUpdates(1000),
        skip_converged=True,
        n_jobs=n_jobs,
        batch_size=1,
        config=parallel_config,
    )
    total_seconds_single_batch = timed_fn.execution_time

    result_multi_batch = timed_fn(
        sampler(u.data.indices, seed=seed),
        u,
        coefficient,
        done=HistoryDeviation(50, 1e-3) | MaxUpdates(1000),
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

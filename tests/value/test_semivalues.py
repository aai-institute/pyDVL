import math
import time
from typing import Type

import numpy as np
import pytest

from pydvl.utils import ParallelConfig, Seed
from pydvl.value.sampler import (
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
from pydvl.value.stopping import AbsoluteStandardError, MaxUpdates

from . import check_values
from .utils import measure_execution_time


@pytest.mark.parametrize("num_samples", [5])
@pytest.mark.parametrize(
    "sampler",
    [
        DeterministicUniformSampler,
        DeterministicPermutationSampler,
        UniformSampler,
        PermutationSampler,
        AntitheticSampler,
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
    criterion = AbsoluteStandardError(0.02, 1.0) | MaxUpdates(2 ** (num_samples * 2))
    values = compute_generic_semivalues(
        sampler(u.data.indices),
        u,
        coefficient,
        criterion,
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
    criterion = AbsoluteStandardError(0.02, 1.0) | MaxUpdates(2 ** (num_samples * 2))
    result_single_batch, total_seconds_single_batch = measure_execution_time(
        compute_generic_semivalues
    )(
        sampler(u.data.indices, seed=seed),
        u,
        coefficient,
        criterion,
        n_jobs=n_jobs,
        batch_size=1,
        config=parallel_config,
    )
    result_multi_batch, total_seconds_multi_batch = measure_execution_time(
        compute_generic_semivalues
    )(
        sampler(u.data.indices, seed=seed),
        u,
        coefficient,
        criterion,
        n_jobs=n_jobs,
        batch_size=batch_size,
        config=parallel_config,
    )
    assert total_seconds_multi_batch < total_seconds_single_batch
    check_values(result_single_batch, result_multi_batch, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("num_samples", [5])
@pytest.mark.parametrize(
    "sampler",
    [
        DeterministicUniformSampler,
        DeterministicPermutationSampler,
        UniformSampler,
        PermutationSampler,
        AntitheticSampler,
    ],
)
def test_banzhaf(
    num_samples: int,
    analytic_banzhaf,
    sampler: Type[PowersetSampler],
    n_jobs: int,
    parallel_config: ParallelConfig,
):
    u, exact_values = analytic_banzhaf
    values = compute_generic_semivalues(
        sampler(u.data.indices),
        u,
        banzhaf_coefficient,
        AbsoluteStandardError(0.04, 1.0) | MaxUpdates(2 ** (num_samples * 2)),
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

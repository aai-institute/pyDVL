import math
from typing import Dict, Type

import numpy as np
import pytest

from pydvl.utils import ParallelConfig, Utility
from pydvl.value.sampler import (
    AntitheticSampler,
    DeterministicPermutationSampler,
    DeterministicUniformSampler,
    PermutationSampler,
    PowersetSampler,
    UniformSampler,
)
from pydvl.value.semivalues import (
    SemiValueMode,
    SVCoefficient,
    banzhaf_coefficient,
    beta_coefficient,
    compute_semivalues,
    semivalues,
    shapley_coefficient,
)
from pydvl.value.stopping import AbsoluteStandardError, MaxUpdates

from . import check_values


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
    criterion = AbsoluteStandardError(0.05, 1.0) | MaxUpdates(2 ** (num_samples * 2))
    values = semivalues(
        sampler(u.data.indices),
        u,
        coefficient,
        criterion,
        n_jobs=n_jobs,
        config=parallel_config,
    )
    check_values(values, exact_values, rtol=0.2)


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
    values = semivalues(
        sampler(u.data.indices),
        u,
        banzhaf_coefficient,
        AbsoluteStandardError(0.05, 1.0) | MaxUpdates(2**10),
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


@pytest.mark.parametrize("num_samples", [5])
@pytest.mark.parametrize(
    "semi_value_mode,semi_value_mode_kwargs",
    [
        (SemiValueMode.Shapley, dict()),
        (SemiValueMode.BetaShapley, {"alpha": 1, "beta": 16}),
        (SemiValueMode.Banzhaf, dict()),
    ],
    ids=["shapley", "beta-shapley", "banzhaf"],
)
def test_dispatch_compute_semi_values(
    dummy_utility: Utility,
    semi_value_mode: SemiValueMode,
    semi_value_mode_kwargs: Dict[str, int],
):
    values = compute_semivalues(
        u=dummy_utility,
        mode=semi_value_mode,
        done=MaxUpdates(1),
        **semi_value_mode_kwargs,
        n_jobs=1,
    )

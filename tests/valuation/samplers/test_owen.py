"""This module tests some functionality specific to owen samplers. Common features
are tested in test_sampler.py"""

from itertools import islice

import numpy as np
import pytest

from pydvl.valuation import (
    AntitheticOwenSampler,
    FiniteSequentialIndexIteration,
    GridOwenStrategy,
    OwenSampler,
)


def _check_sample_sizes(samples, n_samples_outer, n_indices, probs):
    sizes = np.array([len(sample.subset) for sample in samples])
    avg_sizes = sizes.reshape(n_samples_outer, -1).mean(axis=1)
    expected_sizes = probs * n_indices  # mean of Binomial(n_indices, probs)
    np.testing.assert_allclose(avg_sizes, expected_sizes, rtol=0.01, atol=1)


@pytest.mark.flaky(reruns=1)
def test_finite_owen_sampler(seed):
    n_outer = 5
    n_inner = 100
    sampler = OwenSampler(
        outer_sampling_strategy=GridOwenStrategy(n_outer),
        n_samples_inner=n_inner,
        index_iteration=FiniteSequentialIndexIteration,
        seed=seed,
    )
    indices = np.arange(2000)

    # extract samples for the first and second index
    n_samples = n_outer * n_inner
    samples = [b[0] for b in islice(sampler.generate_batches(indices), n_samples * 2)]
    samples_0 = samples[:n_samples]
    samples_1 = samples[n_samples:]

    # check that indices are correct
    assert all(sample.idx == 0 for sample in samples_0)
    assert all(sample.idx == 1 for sample in samples_1)

    # check that the sample_sizes are close to expected sizes
    for samples in [samples_0, samples_1]:
        _check_sample_sizes(
            samples,
            n_samples_outer=n_outer,
            n_indices=len(indices),
            probs=np.array([0.0, 0.25, 0.5, 0.75, 1]),
        )


@pytest.mark.flaky(reruns=1)
def test_antithetic_owen_sampler():
    n_outer = 5
    n_inner = 100
    sampler = AntitheticOwenSampler(
        outer_sampling_strategy=GridOwenStrategy(n_outer),
        n_samples_inner=n_inner,
        index_iteration=FiniteSequentialIndexIteration,
    )
    indices = np.arange(1000)

    # extract samples
    n_samples = n_outer * n_inner * 4
    samples = [b[0] for b in islice(sampler.generate_batches(indices), n_samples)]

    # check that the sample sizes are close to expected sizes
    sizes = np.array([len(sample.subset) for sample in samples])
    avg_sizes = sizes.reshape(n_outer, -1).mean(axis=1)
    np.testing.assert_allclose(avg_sizes, len(indices) // 2, rtol=0.01, atol=1e-5)

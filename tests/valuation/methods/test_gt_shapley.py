from itertools import islice

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pydvl.valuation.methods.gt_shapley import (
    _create_sample_sizes,
    _create_sampling_probabilities,
    compute_n_samples,
)
from pydvl.valuation.samplers import (
    GroupTestingSampleSize,
    NoIndexIteration,
    RandomSizeIteration,
    StratifiedSampler,
)


def test_create_sample_sizes():
    got = _create_sample_sizes(4)
    expected = np.array([1, 2, 3])
    assert_array_equal(got, expected)


def test_create_sampling_probabilities():
    got = _create_sampling_probabilities(np.array([1, 2, 3]))
    expected = np.array([4 / 11, 3 / 11, 4 / 11])
    assert_allclose(got, expected)


def test_compute_n_samples_updated():
    got = compute_n_samples(epsilon=0.1, delta=0.1, n_obs=4)
    # calculated with pen and paper using the formula in Theorem 4 of the updated paper
    expected = 112813
    assert_allclose(got, expected)


def test_gt_sampler(seed):
    sampler = StratifiedSampler(
        index_iteration=NoIndexIteration,
        sample_sizes=GroupTestingSampleSize(n_samples=10),
        sample_sizes_iteration=RandomSizeIteration,
        batch_size=1,
        seed=seed,
    )
    indices = np.arange(10)

    n_samples = 5000

    sizes = []
    for batch in islice(sampler.generate_batches(indices), n_samples):
        sizes.append(len(batch[0].subset))

    uniques, counts = np.unique(sizes, return_counts=True)

    assert len(uniques) == len(indices) - 1

    frequencies = counts / n_samples
    sample_sizes = _create_sample_sizes(len(indices))
    expected = _create_sampling_probabilities(sample_sizes)

    assert_allclose(frequencies, expected, atol=0.02)


def test_gt_setsize_strategy():
    indices = np.arange(10)
    n_samples = 17
    strategy = GroupTestingSampleSize(n_samples=n_samples)
    np.testing.assert_allclose(
        n_samples, strategy.sample_sizes(len(indices), probs=False).sum()
    )

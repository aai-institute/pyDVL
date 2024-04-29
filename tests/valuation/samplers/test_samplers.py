from itertools import takewhile

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pydvl.utils.numeric import powerset
from pydvl.valuation.samplers.permutation import (
    AntitheticPermutationSampler,
    DeterministicPermutationSampler,
    PermutationSampler,
)
from pydvl.valuation.samplers.powerset import (
    AntitheticSampler,
    DeterministicUniformSampler,
    LOOSampler,
    UniformSampler,
    UniformStratifiedSampler,
)


def test_deterministic_uniform_sampler_batch_size_1():
    sampler = DeterministicUniformSampler()
    indices = np.array([1, 2, 3])
    max_iterations = 5

    batches = []
    for batch in sampler.from_indices(indices):
        batches.append(batch)
        if len(batches) >= max_iterations:
            break
    # batches = list(takewhile(lambda _: sampler.n_samples < max_iterations, sampler.from_indices(indices)))
    expected_idxs = [[1], [1], [1], [1], [2]]
    _check_idxs(batches, expected_idxs)

    expected_subsets = [
        [np.array([])],
        [np.array([2])],
        [np.array([3])],
        [np.array([2, 3])],
        [np.array([])],
    ]
    _check_subsets(batches, expected_subsets)


def test_deterministic_uniform_sampler_batch_size_4():
    sampler = DeterministicUniformSampler(batch_size=4)
    indices = np.array([1, 2, 3])
    max_iterations = 8

    batches = list(
        takewhile(
            lambda _: sampler.n_samples < max_iterations, sampler.from_indices(indices)
        )
    )

    expected_idxs = [[1, 1, 1, 1], [2, 2, 2, 2]]
    _check_idxs(batches, expected_idxs)

    expected_subsets = [
        [np.array([]), np.array([2]), np.array([3]), np.array([2, 3])],
        [np.array([]), np.array([1]), np.array([3]), np.array([1, 3])],
    ]
    _check_subsets(batches, expected_subsets)


def test_deterministic_permutation_sampler_batch_size_1():
    sampler = DeterministicPermutationSampler()
    indices = np.array([0, 1, 2])
    max_iterations = 8

    batches = list(
        takewhile(
            lambda _: sampler.n_samples < max_iterations, sampler.from_indices(indices)
        )
    )

    expected_idxs = [[0], [1], [2], [0], [2], [1], [1], [0]]
    _check_idxs(batches, expected_idxs)

    expected_subsets = [
        [np.array([])],
        [np.array([0])],
        [np.array([0, 1])],
        [np.array([])],
        [np.array([0])],
        [np.array([0, 2])],
        [np.array([])],
        [np.array([1])],
    ]
    _check_subsets(batches, expected_subsets)


def test_loo_sampler_batch_size_1():
    sampler = LOOSampler()
    indices = np.array([0, 1, 2])
    max_iterations = 3

    batches = list(
        takewhile(
            lambda _: sampler.n_samples < max_iterations, sampler.from_indices(indices)
        )
    )

    expected_idxs = [[0], [1], [2]]
    _check_idxs(batches, expected_idxs)

    expected_subsets = [
        [np.array([1, 2])],
        [np.array([0, 2])],
        [np.array([0, 1])],
    ]
    _check_subsets(batches, expected_subsets)


def _check_idxs(batches, expected):
    for batch, expected_batch in zip(batches, expected):
        for sample, expected_idx in zip(batch, expected_batch):
            assert sample.idx == expected_idx


def _check_subsets(batches, expected):
    for batch, expected_batch in zip(batches, expected):
        for sample, expected_subset in zip(batch, expected_batch):
            assert_array_equal(sample.subset, expected_subset)


@pytest.mark.parametrize(
    "sampler_class",
    [
        DeterministicUniformSampler,
        UniformSampler,
        DeterministicPermutationSampler,
        PermutationSampler,
        AntitheticSampler,
        UniformStratifiedSampler,
        AntitheticPermutationSampler,
        LOOSampler,
    ],
)
@pytest.mark.parametrize("indices", [np.array([0, 1, 2])])
def test_proper(sampler_class, indices):
    """Test that the sampler generates subsets of the correct sets"""
    sampler = sampler_class()
    max_iterations = 2 ** (len(indices))
    samples = takewhile(
        lambda _: sampler.n_samples < max_iterations, sampler.from_indices(indices)
    )
    for batch in samples:
        sample = list(batch)[0]
        idx, subset = sample
        subsets = [set(s) for s in powerset(np.setxor1d(indices, [idx]))]
        assert set(subset) in subsets


@pytest.mark.parametrize(
    "sampler_class",
    [
        DeterministicUniformSampler,
        UniformSampler,
        DeterministicPermutationSampler,
        PermutationSampler,
        AntitheticSampler,
        UniformStratifiedSampler,
        AntitheticPermutationSampler,
        LOOSampler,
    ],
)
@pytest.mark.parametrize("indices", [np.array([0, 1, 2])])
def test_sample_counter(sampler_class, indices):
    """Test that the sample counter indeed reflects the number of samples generated.

    This test was introduced after finding a bug in the DeterministicUniformSampler
    that was not caused by existing tests.

    """
    sampler = sampler_class()
    max_iterations = 2 ** (len(indices))
    samples = list(
        takewhile(
            lambda _: sampler.n_samples < max_iterations, sampler.from_indices(indices)
        )
    )
    assert sampler.n_samples == len(samples)

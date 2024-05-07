from itertools import takewhile
from typing import Iterator, List, Type, Union

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pydvl.utils import Seed
from pydvl.utils.numeric import powerset
from pydvl.valuation.samplers.permutation import (
    AntitheticPermutationSampler,
    DeterministicPermutationSampler,
    PermutationSampler,
)
from pydvl.valuation.samplers.powerset import (
    AntitheticSampler,
    DeterministicUniformSampler,
    IndexIteration,
    LOOSampler,
    NoIndexIteration,
    PowersetSampler,
    RandomIndexIteration,
    SequentialIndexIteration,
    UniformSampler,
    UniformStratifiedSampler,
)

# TODO Replace by Intersection[StochasticSamplerMixin, PowersetSampler[T]]
# See https://github.com/python/typing/issues/213
StochasticSampler = Union[
    UniformSampler,
    PermutationSampler,
    AntitheticSampler,
    UniformStratifiedSampler,
    AntitheticPermutationSampler,
]


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

    batches = list(sampler.from_indices(indices))

    expected_idxs = [[-1]] * 6
    _check_idxs(batches, expected_idxs)

    expected_subsets = [
        [np.array([0, 1, 2])],
        [np.array([0, 2, 1])],
        [np.array([1, 0, 2])],
        [np.array([1, 2, 0])],
        [np.array([2, 0, 1])],
        [np.array([2, 1, 0])],
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


@pytest.mark.parametrize(
    "sampler, expected_length",
    [
        (DeterministicUniformSampler(), 12),
        (DeterministicUniformSampler(index_iteration=NoIndexIteration), 8),
        (DeterministicPermutationSampler(), 6),
        (LOOSampler(), 3),
    ],
)
def test_length_for_finite_samplers(sampler, expected_length):
    indices = np.array([0, 1, 2])
    assert sampler.length(indices) == expected_length
    assert len(list(sampler.from_indices(indices))) == expected_length


@pytest.mark.parametrize(
    "sampler_class",
    [
        UniformSampler,
        PermutationSampler,
        AntitheticSampler,
        UniformStratifiedSampler,
        AntitheticPermutationSampler,
    ],
)
@pytest.mark.parametrize(
    "index_iteration", [SequentialIndexIteration, RandomIndexIteration]
)
@pytest.mark.parametrize("indices", [(list(range(100)))])
def test_proper_reproducible(sampler_class, index_iteration, indices, seed):
    """Test that the sampler is reproducible."""
    samples_1 = _create_seeded_sample_iter(
        sampler_class, index_iteration, indices, seed
    )
    samples_2 = _create_seeded_sample_iter(
        sampler_class, index_iteration, indices, seed
    )
    for batch_1, batch_2 in zip(samples_1, samples_2):
        assert set(batch_1[0].subset) == set(batch_2[0].subset)


@pytest.mark.parametrize(
    "sampler_class",
    [
        UniformSampler,
        AntitheticSampler,
        UniformStratifiedSampler,
    ],
)
@pytest.mark.parametrize("indices", [(list(range(100)))])
@pytest.mark.parametrize(
    "index_iteration", [SequentialIndexIteration, RandomIndexIteration]
)
def test_proper_stochastic(sampler_class, index_iteration, indices, seed, seed_alt):
    """Test that the sampler is reproducible."""
    samples_1 = _create_seeded_sample_iter(
        sampler_class, index_iteration, indices, seed
    )
    samples_2 = _create_seeded_sample_iter(
        sampler_class, index_iteration, indices, seed_alt
    )

    for batch_1, batch_2 in zip(samples_1, samples_2):
        subset_1 = list(batch_1)[0].subset
        subset_2 = list(batch_2)[0].subset
        assert len(subset_1) == 0 or set(subset_1) != set(subset_2)


def _create_seeded_sample_iter(
    sampler_t: Type[StochasticSampler],
    index_iteration: Type[IndexIteration],
    indices: List,
    seed: Seed,
) -> Iterator:
    max_iterations = len(indices)
    if isinstance(sampler_t, PowersetSampler):
        sampler = sampler_t(index_iteration=index_iteration, seed=seed)
    else:
        sampler = sampler_t(seed=seed)
    sample_stream = takewhile(
        lambda _: sampler.n_samples < max_iterations, sampler.from_indices(indices)
    )
    return sample_stream

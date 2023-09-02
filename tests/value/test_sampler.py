from itertools import takewhile
from typing import Iterator, List, Type

import numpy as np
import pytest

from pydvl.utils import powerset
from pydvl.utils.types import Seed
from pydvl.value.sampler import (
    AntitheticSampler,
    DeterministicPermutationSampler,
    DeterministicUniformSampler,
    PermutationSampler,
    RandomHierarchicalSampler,
    StochasticSampler,
    UniformSampler,
)


@pytest.mark.parametrize(
    "sampler_class",
    [
        DeterministicUniformSampler,
        UniformSampler,
        DeterministicPermutationSampler,
        PermutationSampler,
        AntitheticSampler,
        RandomHierarchicalSampler,
    ],
)
@pytest.mark.parametrize("indices", [(), ([0, 1, 2])])
def test_proper(sampler_class, indices):
    """Test that the sampler generates subsets of the correct sets"""
    sampler = sampler_class(np.array(indices))
    max_iterations = 2 ** (len(indices))
    samples = takewhile(lambda _: sampler.n_samples < max_iterations, sampler)
    for idx, subset in samples:
        subsets = [set(s) for s in powerset(np.setxor1d(indices, [idx]))]
        assert set(subset) in subsets


@pytest.mark.parametrize(
    "sampler_class",
    [
        UniformSampler,
        PermutationSampler,
        AntitheticSampler,
        RandomHierarchicalSampler,
    ],
)
@pytest.mark.parametrize("indices", [(), (list(range(100)))])
def test_proper_reproducible(sampler_class, indices, seed):
    """Test that the sampler is reproducible."""
    samples_1 = _create_seeded_sample_iter(sampler_class, indices, seed)
    samples_2 = _create_seeded_sample_iter(sampler_class, indices, seed)

    for (_, subset_1), (_, subset_2) in zip(samples_1, samples_2):
        assert set(subset_1) == set(subset_2)


@pytest.mark.parametrize(
    "sampler_class",
    [
        UniformSampler,
        PermutationSampler,
        AntitheticSampler,
        RandomHierarchicalSampler,
    ],
)
@pytest.mark.parametrize("indices", [(), (list(range(100)))])
def test_proper_stochastic(sampler_class, indices, seed, seed_alt):
    """Test that the sampler is reproducible."""
    samples_1 = _create_seeded_sample_iter(sampler_class, indices, seed)
    samples_2 = _create_seeded_sample_iter(sampler_class, indices, seed_alt)

    for (_, subset_1), (_, subset_2) in zip(samples_1, samples_2):
        assert len(subset_1) == 0 or set(subset_1) != set(subset_2)


@pytest.mark.parametrize(
    "sampler_class",
    [
        DeterministicUniformSampler,
        UniformSampler,
        #    PermutationSampler,
        AntitheticSampler,
        RandomHierarchicalSampler,
    ],
)
def test_chunkify(sampler_class):
    indices = np.arange(10)
    s1 = sampler_class(indices)
    s2 = s1[:5]
    s3 = s1[5:]
    assert isinstance(s2, sampler_class)
    assert isinstance(s3, sampler_class)
    assert s1._n == s2._n == s3._n
    assert len(s2) == len(s3)
    assert len(s1) == len(s2) + len(s3)


@pytest.mark.parametrize(
    "sampler_class", [DeterministicPermutationSampler, PermutationSampler]
)
def test_chunkify_permutation(sampler_class):
    indices = np.arange(10)
    s1 = sampler_class(indices)
    s2 = s1[:5]
    assert isinstance(s2, sampler_class)
    assert len(s1) == len(s2)


# Missing tests for:
#  - Correct distribution of subsets for random samplers


def _create_seeded_sample_iter(
    sampler_t: Type[StochasticSampler],
    indices: List,
    seed: Seed,
) -> Iterator:
    max_iterations = len(indices)
    sampler = sampler_t(indices=np.array(indices), seed=seed)
    sample_stream = takewhile(lambda _: sampler.n_samples < max_iterations, sampler)
    return sample_stream

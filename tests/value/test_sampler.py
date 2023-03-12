from itertools import takewhile

import numpy as np
import pytest

from pydvl.utils import powerset
from pydvl.value.sampler import (
    AntitheticSampler,
    DeterministicSampler,
    RandomHierarchicalSampler,
    PermutationSampler,
    UniformSampler,
)


@pytest.mark.parametrize(
    "sampler_class",
    [
        DeterministicSampler,
        UniformSampler,
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
        DeterministicSampler,
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


def test_chunkify_permutation():
    indices = np.arange(10)
    s1 = PermutationSampler(indices)
    s2 = s1[:5]
    assert isinstance(s2, PermutationSampler)
    assert len(s1) == len(s2)


# Missing tests for:
#  - Correct distribution of subsets for random samplers

from itertools import islice, takewhile
from typing import Iterator, List, Type, Union

import numpy as np
import pytest

from pydvl.utils import Seed
from pydvl.utils.numeric import powerset
from pydvl.valuation.samplers import (
    AntitheticOwenSampler,
    AntitheticSampler,
    DeterministicUniformSampler,
    IndexIteration,
    LOOSampler,
    MSRSampler,
    NoIndexIteration,
    OwenSampler,
    PowersetSampler,
    RandomIndexIteration,
    SequentialIndexIteration,
    TruncatedUniformStratifiedSampler,
    UniformSampler,
    UniformStratifiedSampler,
    VarianceReducedStratifiedSampler,
)
from pydvl.valuation.samplers.permutation import (
    AntitheticPermutationSampler,
    DeterministicPermutationSampler,
    PermutationSampler,
)

from . import _check_idxs, _check_subsets

# TODO Replace by Intersection[StochasticSamplerMixin, PowersetSampler[T]]
# See https://github.com/python/typing/issues/213
StochasticSampler = Union[
    UniformSampler,
    PermutationSampler,
    AntitheticSampler,
    UniformStratifiedSampler,
    AntitheticPermutationSampler,
    MSRSampler,
]


def test_deterministic_uniform_sampler_batch_size_1():
    sampler = DeterministicUniformSampler()
    indices = np.array([1, 2, 3])
    max_iterations = 5

    batches = list(
        takewhile(
            lambda _: sampler.n_samples < max_iterations,
            sampler.generate_batches(indices),
        )
    )
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
            lambda _: sampler.n_samples < max_iterations,
            sampler.generate_batches(indices),
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

    batches = list(sampler.generate_batches(indices))

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
            lambda _: sampler.n_samples < max_iterations,
            sampler.generate_batches(indices),
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


@pytest.mark.parametrize(
    "sampler",
    [
        DeterministicUniformSampler(),
        UniformSampler(),
        DeterministicPermutationSampler(),
        PermutationSampler(),
        AntitheticSampler(),
        UniformStratifiedSampler(),
        AntitheticPermutationSampler(),
        LOOSampler(),
        UniformSampler(index_iteration=RandomIndexIteration),
        UniformStratifiedSampler(index_iteration=RandomIndexIteration),
        AntitheticSampler(index_iteration=RandomIndexIteration),
        TruncatedUniformStratifiedSampler(lower_bound=1, upper_bound=2),
        VarianceReducedStratifiedSampler(samples_per_setsize=lambda _: 2),
        OwenSampler(n_samples_outer=4),
        AntitheticOwenSampler(n_samples_outer=2),
        MSRSampler(),
    ],
)
@pytest.mark.parametrize("indices", [np.array([]), np.array([0, 1, 2])])
def test_proper(sampler, indices):
    """Test that the sampler generates subsets of the correct sets"""
    max_iterations = 2 ** (len(indices))
    samples = takewhile(
        lambda _: sampler.n_samples < max_iterations, sampler.generate_batches(indices)
    )
    for batch in samples:
        sample = list(batch)[0]
        idx, subset = sample
        if idx is not None:
            subsets = [set(s) for s in powerset(np.setxor1d(indices, [idx]))]
        else:
            subsets = [set(s) for s in powerset(indices)]
        assert set(subset) in subsets


@pytest.mark.parametrize(
    "sampler",
    [
        DeterministicUniformSampler(),
        UniformSampler(),
        DeterministicPermutationSampler(),
        PermutationSampler(),
        AntitheticSampler(),
        UniformStratifiedSampler(),
        AntitheticPermutationSampler(),
        LOOSampler(),
        UniformSampler(index_iteration=RandomIndexIteration),
        UniformStratifiedSampler(index_iteration=RandomIndexIteration),
        AntitheticSampler(index_iteration=RandomIndexIteration),
        TruncatedUniformStratifiedSampler(lower_bound=1, upper_bound=2),
        VarianceReducedStratifiedSampler(samples_per_setsize=lambda _: 2),
        OwenSampler(n_samples_outer=4),
        AntitheticOwenSampler(n_samples_outer=2),
        MSRSampler(),
    ],
)
def test_sample_counter(sampler):
    """Test that the sample counter indeed reflects the number of samples generated.

    This test was introduced after finding a bug in the DeterministicUniformSampler
    that was not caused by existing tests.

    """
    indices = np.array([0, 1, 2])
    max_iterations = 2 ** (len(indices))
    samples = list(
        takewhile(
            lambda _: sampler.n_samples < max_iterations,
            sampler.generate_batches(indices),
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
        (OwenSampler(n_samples_outer=4, n_samples_inner=2), 4 * 2 * 3),
        (AntitheticOwenSampler(n_samples_outer=3, n_samples_inner=2), 2 * 3 * 2 * 3),
    ],
)
def test_length_for_finite_samplers(sampler, expected_length):
    indices = np.array([0, 1, 2])
    assert sampler.sample_limit(indices) == expected_length
    assert len(list(sampler.generate_batches(indices))) == expected_length


@pytest.mark.parametrize(
    "sampler",
    [
        UniformSampler(),
        PermutationSampler(),
        AntitheticSampler(),
        UniformStratifiedSampler(),
        AntitheticPermutationSampler(),
        TruncatedUniformStratifiedSampler(lower_bound=1, upper_bound=2),
        VarianceReducedStratifiedSampler(samples_per_setsize=lambda _: 2),
        MSRSampler(),
    ],
)
def test_length_of_infinite_samplers(sampler):
    indices = np.array([0, 1, 2])
    max_iter = 2 ** len(indices) * 10
    assert sampler.sample_limit(indices) is None
    # check that we can generate samples that are longer than size of powerset
    samples = list(
        takewhile(
            lambda _: sampler.n_samples < max_iter, sampler.generate_batches(indices)
        )
    )
    assert len(samples) == max_iter


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
@pytest.mark.parametrize("indices", [np.array([]), np.array(list(range(100)))])
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
@pytest.mark.parametrize("indices", [np.array([]), np.array(list(range(100)))])
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
    if issubclass(sampler_t, PowersetSampler):
        sampler = sampler_t(index_iteration=index_iteration, seed=seed)
    else:
        sampler = sampler_t(seed=seed)
    sample_stream = takewhile(
        lambda _: sampler.n_samples < max_iterations, sampler.generate_batches(indices)
    )
    return sample_stream


@pytest.mark.flaky(reruns=1)
def test_owen_sampler():
    n_outer = 5
    n_inner = 100
    sampler = OwenSampler(
        n_samples_outer=n_outer, n_samples_inner=n_inner, batch_size=1
    )
    indices = np.arange(5000)

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
    n_outer = 3
    n_inner = 100
    sampler = AntitheticOwenSampler(
        n_samples_outer=n_outer, n_samples_inner=n_inner, batch_size=1
    )
    indices = np.arange(5000)

    # extract samples for the first and second index
    n_samples = n_outer * n_inner
    samples = [b[0] for b in islice(sampler.generate_batches(indices), n_samples * 4)]
    samples_0 = samples[: n_samples * 2 : 2]
    samples_0_complement = samples[1 : 2 * n_samples : 2]
    samples_1 = samples[2 * n_samples :: 2]
    samples_1_complement = samples[2 * n_samples + 1 :: 2]

    # check that indices are correct
    assert all(sample.idx == 0 for sample in samples_0)
    assert all(sample.idx == 0 for sample in samples_0_complement)
    assert all(sample.idx == 1 for sample in samples_1)
    assert all(sample.idx == 1 for sample in samples_1_complement)

    # check that the sample_sizes are close to expected sizes
    for samples in [samples_0, samples_1]:
        _check_sample_sizes(
            samples,
            n_samples_outer=n_outer,
            n_indices=len(indices),
            probs=np.array([0.0, 0.25, 0.5]),
        )
    for samples in [samples_0_complement, samples_1_complement]:
        _check_sample_sizes(
            samples,
            n_samples_outer=n_outer,
            n_indices=len(indices),
            probs=np.array([1.0, 0.75, 0.5]),
        )


def _check_sample_sizes(samples, n_samples_outer, n_indices, probs):
    sizes = np.array([len(sample.subset) for sample in samples])
    avg_sizes = sizes.reshape(n_samples_outer, -1).mean(axis=1)
    expected_sizes = probs * n_indices
    assert np.allclose(avg_sizes, expected_sizes, rtol=0.01)

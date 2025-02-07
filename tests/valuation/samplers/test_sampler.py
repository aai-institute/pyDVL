import math
from itertools import islice, takewhile
from typing import Any, Iterator, Type

import numpy as np
import pytest
from more_itertools import flatten
from numpy.typing import NDArray

from pydvl.utils.numeric import powerset
from pydvl.utils.types import Seed
from pydvl.valuation import (
    HarmonicSamplesPerSetSize,
    IndexSampler,
    PowerLawSamplesPerSetSize,
)
from pydvl.valuation.samplers import (
    AntitheticOwenSampler,
    AntitheticPermutationSampler,
    AntitheticSampler,
    DeterministicPermutationSampler,
    DeterministicUniformSampler,
    FiniteOwenSampler,
    IndexIteration,
    LOOSampler,
    MSRSampler,
    NoIndexIteration,
    OwenSampler,
    PermutationSampler,
    PowersetSampler,
    RandomIndexIteration,
    SequentialIndexIteration,
    StochasticSampler,
    TruncatedUniformStratifiedSampler,
    UniformSampler,
    UniformStratifiedSampler,
    VarianceReducedStratifiedSampler,
)

from .. import recursive_make
from . import _check_idxs, _check_subsets

DETERMINISTIC_SAMPLERS: list[tuple[Type, dict]] = [
    (DeterministicUniformSampler, {}),
    (DeterministicPermutationSampler, {}),
    (LOOSampler, {}),
]


RANDOM_SAMPLERS: list[tuple[Type, dict]] = [
    (UniformSampler, {}),
    (AntitheticSampler, {}),
    (PermutationSampler, {}),
    (AntitheticPermutationSampler, {}),
    (UniformStratifiedSampler, {}),
    (UniformStratifiedSampler, {}),
    (TruncatedUniformStratifiedSampler, {"lower_bound": 2, "upper_bound": 3}),
    (
        VarianceReducedStratifiedSampler,
        {
            "samples_per_setsize": (
                HarmonicSamplesPerSetSize,
                {"n_samples_per_index": 32},
            )
        },
    ),
    (FiniteOwenSampler, {"n_samples_outer": 4}),
    (OwenSampler, {}),
    (AntitheticOwenSampler, {}),
    (MSRSampler, {}),
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
    "sampler_cls, sampler_kwargs", DETERMINISTIC_SAMPLERS + RANDOM_SAMPLERS
)
@pytest.mark.parametrize("indices", [np.array([]), np.arange(5)])
def test_proper(sampler_cls, sampler_kwargs, indices):
    """Test that the sampler generates subsets of the correct sets"""
    sampler = recursive_make(sampler_cls, sampler_kwargs)
    max_iterations = 2 ** (len(indices))
    samples = takewhile(
        lambda _: sampler.n_samples < max_iterations, sampler.generate_batches(indices)
    )
    for batch in samples:
        sample = list(batch)[0]
        idx, subset = sample
        subsets = [set(s) for s in powerset(np.setdiff1d(indices, [idx]))]
        assert set(subset) in subsets


@pytest.mark.parametrize(
    "sampler_cls, sampler_kwargs", DETERMINISTIC_SAMPLERS + RANDOM_SAMPLERS
)
def test_sample_counter(sampler_cls, sampler_kwargs):
    """Test that the sample counter indeed reflects the number of samples generated.

    This test was introduced after finding a bug in the DeterministicUniformSampler
    that was not caused by existing tests.
    """
    sampler = recursive_make(sampler_cls, sampler_kwargs)
    indices = np.arange(4)
    max_iterations = 2 ** (len(indices) + 1)
    samples = list(
        takewhile(
            lambda _: sampler.n_samples < max_iterations,
            sampler.generate_batches(indices),
        )
    )
    assert sampler.n_samples == len(list(flatten(samples)))


@pytest.mark.parametrize("indices", [np.array([]), np.arange(3)])
@pytest.mark.parametrize(
    "sampler_cls, sampler_kwargs, expected_length",
    [
        (DeterministicUniformSampler, {}, lambda n: n * 4),
        (
            DeterministicUniformSampler,
            {"index_iteration": NoIndexIteration},
            lambda n: 2**n if n > 0 else 0,
        ),
        (
            DeterministicPermutationSampler,
            {},
            lambda n: math.factorial(n) if n > 0 else 0,
        ),
        (LOOSampler, {}, lambda n: n),
        (
            FiniteOwenSampler,
            {"n_samples_outer": 4, "n_samples_inner": 2},
            lambda n: 4 * 2 * n,
        ),
        (
            VarianceReducedStratifiedSampler,
            {
                "samples_per_setsize": (
                    HarmonicSamplesPerSetSize,
                    {"n_samples_per_index": 32},
                )
            },
            lambda n: n * 32,
        ),
        (
            VarianceReducedStratifiedSampler,
            {
                "samples_per_setsize": (
                    PowerLawSamplesPerSetSize,
                    {"n_samples_per_index": 13, "exponent": -0.5},
                )
            },
            lambda n: n * 13,
        ),
    ],
)
def test_length_for_finite_samplers(
    indices, sampler_cls, sampler_kwargs, expected_length
):
    sampler = recursive_make(sampler_cls, sampler_kwargs)
    assert sampler.sample_limit(indices) == expected_length(len(indices))
    assert len(list(sampler.generate_batches(indices))) == expected_length(len(indices))


@pytest.mark.parametrize("sampler_cls, sampler_kwargs", RANDOM_SAMPLERS)
def test_length_of_infinite_samplers(sampler_cls, sampler_kwargs):
    """All infinite samplers are random, but not all random are infinite..."""
    if sampler_cls in (FiniteOwenSampler, VarianceReducedStratifiedSampler):
        pytest.skip(f"{sampler_cls.__name__} is a finite sampler")
    indices = np.arange(4)
    max_iter = 2 ** len(indices) * 10
    sampler = recursive_make(sampler_cls, sampler_kwargs)
    assert sampler.sample_limit(indices) is None
    # check that we can generate samples that are longer than size of powerset
    samples = list(
        takewhile(
            lambda _: sampler.n_samples < max_iter, sampler.generate_batches(indices)
        )
    )
    assert len(samples) == max_iter


@pytest.mark.parametrize("sampler_cls, sampler_kwargs", RANDOM_SAMPLERS)
@pytest.mark.parametrize(
    "index_iteration", [SequentialIndexIteration, RandomIndexIteration]
)
@pytest.mark.parametrize("indices", [np.array([]), np.array(list(range(100)))])
def test_proper_reproducible(
    sampler_cls, sampler_kwargs, index_iteration, indices, seed
):
    """Test that the sampler is reproducible."""
    samples_1 = _create_seeded_sample_iter(
        sampler_cls, sampler_kwargs, index_iteration, indices, seed
    )
    samples_2 = _create_seeded_sample_iter(
        sampler_cls, sampler_kwargs, index_iteration, indices, seed
    )
    for batch_1, batch_2 in zip(samples_1, samples_2):
        assert set(batch_1[0].subset) == set(batch_2[0].subset)


@pytest.mark.parametrize("sampler_cls, sampler_kwargs", RANDOM_SAMPLERS)
@pytest.mark.parametrize("indices", [np.array([]), np.arange(10)])
@pytest.mark.parametrize(
    "index_iteration", [SequentialIndexIteration, RandomIndexIteration]
)
def test_proper_stochastic(
    sampler_cls, sampler_kwargs, index_iteration, indices, seed, seed_alt
):
    """Test that the sampler is reproducible."""
    samples_1 = _create_seeded_sample_iter(
        sampler_cls, sampler_kwargs, index_iteration, indices, seed
    )
    samples_2 = _create_seeded_sample_iter(
        sampler_cls, sampler_kwargs, index_iteration, indices, seed_alt
    )

    for batch_1, batch_2 in zip(samples_1, samples_2):
        subset_1 = list(batch_1)[0].subset
        subset_2 = list(batch_2)[0].subset
        if issubclass(sampler_cls, PermutationSampler):
            # Order matters for permutations!
            assert len(subset_1) == 0 or np.any(subset_1 != subset_2)
        else:
            assert len(subset_1) == 0 or set(subset_1) != set(subset_2)


def _create_seeded_sample_iter(
    sampler_t: Type[StochasticSampler],
    sampler_kwargs: dict[str, Any],
    index_iteration: Type[IndexIteration],
    indices: NDArray[np.int_],
    seed: Seed,
) -> Iterator:
    sampler: IndexSampler
    # If we set max_iterations to len(indices), then the FiniteOwenSampler will
    # always generate the full sample as last one, failing test_proper_stochastic()
    max_iterations = len(indices) // 2
    if issubclass(sampler_t, PowersetSampler):
        sampler = sampler_t(
            index_iteration=index_iteration, seed=seed, **sampler_kwargs
        )
    else:
        sampler = sampler_t(seed=seed, **sampler_kwargs)
    sample_stream = takewhile(
        lambda _: sampler.n_samples < max_iterations, sampler.generate_batches(indices)
    )
    return sample_stream


@pytest.mark.flaky(reruns=1)
def test_finite_owen_sampler():
    n_outer = 5
    n_inner = 100
    sampler = FiniteOwenSampler(
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
    sampler = AntitheticOwenSampler(n_samples_inner=n_inner, batch_size=1)
    indices = np.arange(5000)

    # extract samples
    n_samples = n_outer * n_inner
    samples = [b[0] for b in islice(sampler.generate_batches(indices), n_samples * 4)]

    # check that the sample sizes are close to expected sizes
    sizes = np.array([len(sample.subset) for sample in samples])
    avg_sizes = sizes.reshape(n_outer, -1).mean(axis=1)
    assert np.allclose(avg_sizes, len(indices) // 2, rtol=0.01)


def _check_sample_sizes(samples, n_samples_outer, n_indices, probs):
    sizes = np.array([len(sample.subset) for sample in samples])
    avg_sizes = sizes.reshape(n_samples_outer, -1).mean(axis=1)
    expected_sizes = probs * n_indices
    assert np.allclose(avg_sizes, expected_sizes, rtol=0.01)


@pytest.mark.parametrize(
    "sampler_cls, sampler_kwargs", DETERMINISTIC_SAMPLERS + RANDOM_SAMPLERS
)
@pytest.mark.parametrize("indices", [np.arange(4)])
def test_sampler_weights(sampler_cls, sampler_kwargs, indices):
    """Test whether weight(n, k) corresponds to the probability of sampled subsets of
    size k from n indices."""

    if issubclass(sampler_cls, LOOSampler):
        pytest.skip("LOOSampler only samples sets of size n-1")

    # Sample 2**n subsets and count the number of subsets of each size
    n = len(indices)
    n_batches = 2 ** (2 * n)  # go high to be sure

    # These samplers return samples up to the size of the whole index set
    if issubclass(sampler_cls, (PermutationSampler, MSRSampler)):
        subset_frequencies = np.zeros(n + 1)
        n += 1
    else:
        subset_frequencies = np.zeros(n)
        sampler_kwargs["batch_size"] = n

    sampler = recursive_make(sampler_cls, sampler_kwargs)
    for batch in islice(sampler.generate_batches(indices), n_batches):
        for sample in batch:
            subset_frequencies[len(sample.subset)] += 1
    subset_frequencies /= subset_frequencies.sum()

    expected_frequencies = np.zeros(n)
    for k in range(n):
        try:
            # Recall that sampler.weight = inverse probability of sampling
            expected_frequencies[k] = math.comb(n - 1, k) / sampler.weight(n, k)
        except ValueError:  # out of bounds in stratified samplers
            pass

    assert np.allclose(subset_frequencies, expected_frequencies, atol=0.05)


@pytest.mark.parametrize(
    "lower_bound, upper_bound, expected_message",
    [
        (-1, None, r"Lower bound"),
        (11, None, r"Lower bound"),
        (5, 4, r"Upper bound"),
        (None, 11, r"Upper bound"),
    ],
)
def test_truncateduniformstratifiedsampler_raises(
    lower_bound, upper_bound, expected_message
):
    indices = np.arange(10)
    with pytest.raises(ValueError, match=expected_message):
        next(
            TruncatedUniformStratifiedSampler(
                lower_bound=lower_bound, upper_bound=upper_bound
            )._generate(indices)
        )

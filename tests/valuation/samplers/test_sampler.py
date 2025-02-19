from __future__ import annotations

import math
from itertools import islice, takewhile
from typing import Any, Iterator, Type

import numpy as np
import pytest
from more_itertools import flatten
from numpy.typing import NDArray

from pydvl.utils.numeric import logcomb, powerset
from pydvl.utils.types import Seed
from pydvl.valuation.samplers import (
    AntitheticOwenSampler,
    AntitheticPermutationSampler,
    AntitheticSampler,
    ConstantSampleSize,
    DeterministicPermutationSampler,
    DeterministicSizeIteration,
    DeterministicUniformSampler,
    FiniteNoIndexIteration,
    FiniteRandomIndexIteration,
    FiniteSequentialIndexIteration,
    GridOwenStrategy,
    HarmonicSampleSize,
    IndexIteration,
    LOOSampler,
    MSRSampler,
    NoIndexIteration,
    OwenSampler,
    PermutationSampler,
    PowerLawSampleSize,
    PowersetSampler,
    RandomIndexIteration,
    RandomSizeIteration,
    SequentialIndexIteration,
    StochasticSampler,
    StratifiedSampler,
    UniformOwenStrategy,
    UniformSampler,
)
from pydvl.valuation.samplers.permutation import PermutationSamplerBase
from pydvl.valuation.types import IndexSetT

from .. import recursive_make
from . import _check_idxs, _check_subsets


def deterministic_samplers():
    return [
        (DeterministicUniformSampler, {}),
        (DeterministicPermutationSampler, {}),
        (LOOSampler, {}),
    ]


def random_samplers(proper: bool = False):
    """Use this as parameter values in pytest.mark.parametrize for parameters
    "sampler_cls, sampler_kwargs"

    Build the objects with recursive_make(sampler_cls, sampler_kwargs, **lambda_args)
    where lambda args are named as the key in the dictionary that contains the lambda.
    """

    improper_samplers = [
        (
            OwenSampler,
            {
                "outer_sampling_strategy": (
                    GridOwenStrategy,
                    {"n_samples_outer": lambda n=200: n},
                ),
                "index_iteration": NoIndexIteration,
            },
        ),
        (
            AntitheticOwenSampler,
            {
                "outer_sampling_strategy": (
                    GridOwenStrategy,
                    {"n_samples_outer": lambda n=200: n},
                ),
                "index_iteration": NoIndexIteration,
            },
        ),
        (
            OwenSampler,
            {
                "outer_sampling_strategy": (
                    GridOwenStrategy,
                    {"n_samples_outer": lambda n=200: n},
                ),
                "index_iteration": FiniteNoIndexIteration,
            },
        ),
        (
            AntitheticOwenSampler,
            {
                "outer_sampling_strategy": (
                    GridOwenStrategy,
                    {"n_samples_outer": lambda n=200: n},
                ),
                "index_iteration": FiniteNoIndexIteration,
            },
        ),
        (
            OwenSampler,
            {
                "outer_sampling_strategy": (
                    UniformOwenStrategy,
                    {"n_samples_outer": lambda n=200: n},
                ),
                "index_iteration": NoIndexIteration,
            },
        ),
        (
            AntitheticOwenSampler,
            {
                "outer_sampling_strategy": (
                    UniformOwenStrategy,
                    {"n_samples_outer": lambda n=200: n},
                ),
                "index_iteration": NoIndexIteration,
            },
        ),
        (
            OwenSampler,
            {
                "outer_sampling_strategy": (
                    UniformOwenStrategy,
                    {"n_samples_outer": lambda n=200: n},
                ),
                "index_iteration": FiniteNoIndexIteration,
            },
        ),
        (
            AntitheticOwenSampler,
            {
                "outer_sampling_strategy": (
                    UniformOwenStrategy,
                    {"n_samples_outer": lambda n=200: n},
                ),
                "index_iteration": FiniteNoIndexIteration,
            },
        ),
    ]

    permutation_samplers = [
        (PermutationSampler, {"seed": lambda seed: seed}),
        (AntitheticPermutationSampler, {"seed": lambda seed: seed}),
    ]

    powerset_samplers = [
        (
            UniformSampler,
            {"index_iteration": RandomIndexIteration, "seed": lambda seed: seed},
        ),
        (
            UniformSampler,
            {"index_iteration": SequentialIndexIteration, "seed": lambda seed: seed},
        ),
        (
            AntitheticSampler,
            {"index_iteration": RandomIndexIteration, "seed": lambda seed: seed},
        ),
        (
            AntitheticSampler,
            {"index_iteration": SequentialIndexIteration, "seed": lambda seed: seed},
        ),
    ]

    stratified_samplers = [
        (
            StratifiedSampler,
            {
                "sample_sizes": (
                    ConstantSampleSize,
                    {
                        "n_samples": lambda n=32: n,
                        "lower_bound": lambda l=2: l,
                        "upper_bound": lambda u=3: u,
                    },
                ),
                "sample_sizes_iteration": RandomSizeIteration,
                "index_iteration": RandomIndexIteration,
            },
        ),
        (
            StratifiedSampler,
            {
                "sample_sizes": (ConstantSampleSize, {"n_samples": lambda n=32: n}),
                "sample_sizes_iteration": RandomSizeIteration,
                "index_iteration": RandomIndexIteration,
            },
        ),
        (
            StratifiedSampler,
            {
                "sample_sizes": (HarmonicSampleSize, {"n_samples": lambda n=32: n}),
                "sample_sizes_iteration": DeterministicSizeIteration,
                "index_iteration": RandomIndexIteration,
            },
        ),
        (
            StratifiedSampler,
            {
                "sample_sizes": (HarmonicSampleSize, {"n_samples": lambda n=32: n}),
                "sample_sizes_iteration": RandomSizeIteration,
                "index_iteration": SequentialIndexIteration,
            },
        ),
        (
            StratifiedSampler,
            {
                "sample_sizes": (
                    PowerLawSampleSize,
                    {"n_samples": lambda n=32: n, "exponent": lambda e=0.5: e},
                ),
                "index_iteration": RandomIndexIteration,
            },
        ),
        (
            StratifiedSampler,
            {
                "sample_sizes": (
                    PowerLawSampleSize,
                    {"n_samples": lambda n=32: n, "exponent": lambda e=0.5: e},
                ),
                "index_iteration": SequentialIndexIteration,
            },
        ),
    ]

    owen_samplers = [
        (
            OwenSampler,
            {
                "outer_sampling_strategy": (
                    GridOwenStrategy,
                    {"n_samples_outer": lambda n=200: n},
                ),
                "index_iteration": FiniteSequentialIndexIteration,
            },
        ),
        (
            OwenSampler,
            {
                "outer_sampling_strategy": (
                    UniformOwenStrategy,
                    {"n_samples_outer": lambda n=200: n, "seed": lambda seed: seed},
                ),
                "index_iteration": FiniteSequentialIndexIteration,
            },
        ),
        (
            AntitheticOwenSampler,
            {
                "outer_sampling_strategy": (
                    GridOwenStrategy,
                    {"n_samples_outer": lambda n=100: n},
                ),
                # required for test_proper_stochastic: with q=0 the 2nd sample is always
                # the full complement and the randomness test fails.
                "batch_size": 2,
                "index_iteration": FiniteSequentialIndexIteration,
            },
        ),
        (
            AntitheticOwenSampler,
            {
                "outer_sampling_strategy": (
                    UniformOwenStrategy,
                    {"n_samples_outer": lambda n=100: n, "seed": lambda seed: seed},
                ),
                "index_iteration": FiniteSequentialIndexIteration,
            },
        ),
        (MSRSampler, {"seed": lambda seed: seed}),
    ]

    return (
        permutation_samplers
        + powerset_samplers
        + stratified_samplers
        + owen_samplers
        + (improper_samplers if not proper else [])
    )


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

    expected_idxs = [[None]] * 6
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


@pytest.mark.parametrize("sampler_cls, sampler_kwargs", random_samplers())
@pytest.mark.parametrize("indices", [np.array([]), np.arange(5)])
def test_proper(sampler_cls, sampler_kwargs: dict, indices: NDArray[np.int_], seed):
    """Test that the sampler generates subsets of the correct sets"""
    sampler = recursive_make(sampler_cls, sampler_kwargs, seed=seed)
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
    "sampler_cls, sampler_kwargs", deterministic_samplers() + random_samplers()
)
def test_sample_counter(sampler_cls, sampler_kwargs: dict, seed: int):
    """Test that the sample counter indeed reflects the number of samples generated.

    This test was introduced after finding a bug in the DeterministicUniformSampler
    that was not caused by existing tests.
    """
    sampler = recursive_make(sampler_cls, sampler_kwargs, seed=seed)
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
        (DeterministicUniformSampler, {}, lambda n: n * 2 ** (n - 1)),
        (
            DeterministicUniformSampler,
            {"index_iteration": FiniteNoIndexIteration},
            lambda n: 2**n if n > 0 else 0,
        ),
        (
            DeterministicPermutationSampler,
            {},
            lambda n: math.factorial(n) if n > 0 else 0,
        ),
        (LOOSampler, {}, lambda n: n),
        # (
        #     StratifiedSampler,
        #     {
        #         "index_iteration": FiniteSequentialIndexIteration,
        #         "sample_sizes": (HarmonicSampleSizes, {"n_samples": 32}),
        #     },
        #     lambda n: TODO: compute this...
        # ),
        (
            OwenSampler,
            {
                "outer_sampling_strategy": (GridOwenStrategy, {"n_samples_outer": 4}),
                "n_samples_inner": 2,
            },
            lambda n: 4 * 2 * n,
        ),
        (
            AntitheticOwenSampler,
            {
                "outer_sampling_strategy": (GridOwenStrategy, {"n_samples_outer": 4}),
                "index_iteration": FiniteSequentialIndexIteration,
            },
            lambda n: 2 * 4 * 2 * n,
        ),
        (
            StratifiedSampler,
            {
                "sample_sizes": (
                    HarmonicSampleSize,
                    {"n_samples": 32},
                )
            },
            lambda n: n * 32,
        ),
        (
            StratifiedSampler,
            {
                "sample_sizes": (
                    PowerLawSampleSize,
                    {"n_samples": 13, "exponent": -0.5},
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


@pytest.mark.parametrize("sampler_cls, sampler_kwargs", random_samplers())
def test_length_of_infinite_samplers(sampler_cls, sampler_kwargs, seed):
    """All infinite samplers are random, but not all random are infinite..."""
    if sampler_cls in (
        OwenSampler,
        AntitheticOwenSampler,
        StratifiedSampler,
    ):
        pytest.skip(f"{sampler_cls.__name__} is a finite sampler")
    indices = np.arange(4)
    max_iter = 2 ** len(indices) * 10
    sampler = recursive_make(sampler_cls, sampler_kwargs, seed=seed)
    assert sampler.sample_limit(indices) is None
    # check that we can generate samples that are longer than size of powerset
    samples = list(
        takewhile(
            lambda _: sampler.n_samples < max_iter, sampler.generate_batches(indices)
        )
    )
    assert len(samples) == max_iter


@pytest.mark.parametrize("sampler_cls, sampler_kwargs", random_samplers(proper=True))
@pytest.mark.parametrize(
    "index_iteration", [FiniteSequentialIndexIteration, FiniteRandomIndexIteration]
)
@pytest.mark.parametrize("indices", [np.array([]), np.array(list(range(100)))])
def test_proper_reproducible(
    sampler_cls, sampler_kwargs: dict, index_iteration, indices: NDArray, seed
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


@pytest.mark.parametrize("sampler_cls, sampler_kwargs", random_samplers(proper=True))
@pytest.mark.parametrize("indices", [np.array([]), np.arange(10)])
@pytest.mark.parametrize(
    "index_iteration", [FiniteSequentialIndexIteration, FiniteRandomIndexIteration]
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
    sampler_cls: Type[StochasticSampler],
    sampler_kwargs: dict[str, Any],
    index_iteration: Type[IndexIteration],
    indices: NDArray[np.int_],
    seed: Seed,
) -> Iterator:
    sampler_kwargs["seed"] = seed
    sampler_kwargs["index_iteration"] = index_iteration
    try:
        sampler = recursive_make(sampler_cls, sampler_kwargs, seed=seed)
    except TypeError:
        del sampler_kwargs["index_iteration"]
        sampler = recursive_make(sampler_cls, sampler_kwargs, seed=seed)

    # If we set max_iterations to len(indices), then the OwenSampler will
    # always generate the full sample as last once, failing test_proper_stochastic()
    max_iterations = len(indices) // 2
    sample_stream = takewhile(
        lambda _: sampler.n_samples < max_iterations, sampler.generate_batches(indices)
    )
    return sample_stream


@pytest.mark.flaky(reruns=1)
@pytest.mark.parametrize(
    "sampler_cls, sampler_kwargs", deterministic_samplers() + random_samplers()
)
@pytest.mark.parametrize("indices", [np.arange(6)])
def test_sampler_weights(
    sampler_cls, sampler_kwargs: dict, indices: NDArray, seed: int
):
    """Test whether weight(n, k) corresponds to the probability of sampling a given
    subset of size k from n indices. Note this is *NOT* P(|S| = k), but instead
    the probability P(S), which depends only on n and k due to the symmetry in the
    sampling process.
    """

    if issubclass(sampler_cls, LOOSampler):
        pytest.skip("LOOSampler only samples sets of size n-1")
    if issubclass(sampler_cls, PermutationSamplerBase):
        pytest.skip("Permutation samplers only sample full sets")

    # Sample and count the number of subsets of each size
    n = len(indices)
    n_batches = 2 ** (n * 2)  # go high to be sure

    sampler = recursive_make(sampler_cls, sampler_kwargs, seed=seed)

    # These samplers return samples up to the size of the whole index set
    if issubclass(sampler_cls, MSRSampler) or issubclass(
        getattr(sampler, "_index_iterator_cls", IndexIteration), NoIndexIteration
    ):
        complement_size = n
        max_size = n + 1
        subset_len_probs = np.zeros(n + 1)
    else:
        complement_size = n - 1
        max_size = n
        subset_len_probs = np.zeros(n)

    # HACK: MSR actually has same distribution as uniform over 2^(N-i)
    fudge = 0.5 if issubclass(sampler_cls, MSRSampler) else 1.0

    for batch in islice(sampler.generate_batches(indices), n_batches):
        for sample in batch:
            subset_len_probs[len(sample.subset)] += 1
    subset_len_probs /= subset_len_probs.sum()

    expected_log_subset_len_probs = np.full(max_size, -np.inf)
    for k in range(max_size):
        try:
            # log_weight = log probability of sampling
            # So: no. of sets of size k in the powerset, times. prob of sampling size k
            expected_log_subset_len_probs[k] = (
                logcomb(complement_size, k) + sampler.log_weight(n, k) + math.log(fudge)
            )
        except ValueError:  # out of bounds in stratified samplers
            pass

    np.testing.assert_allclose(
        subset_len_probs, np.exp(expected_log_subset_len_probs), atol=0.05
    )


class TestSampler(PowersetSampler):
    def __init__(self):
        super().__init__(batch_size=1, index_iteration=FiniteSequentialIndexIteration)

    def _generate(self, indices: IndexSetT):
        pass

    def sample_limit(self, indices: IndexSetT) -> int | None:
        pass


@pytest.mark.parametrize(
    "indices, skip, expected",
    [
        (np.arange(6), np.array([2, 4]), [0, 1, 3, 5]),
        (np.arange(6), np.empty(0), np.arange(6)),
        (np.empty(0), np.arange(6), np.empty(0)),
    ],
)
def test_skip_indices(indices, skip, expected):
    sampler = TestSampler()
    sampler.skip_indices = skip

    result = list(sampler.index_iterator(indices))

    assert set(result) == set(expected), f"Expected {expected}, but got {result}"


def test_skip_indices_after_first_batch():
    n_indices = 4
    indices = np.arange(n_indices)
    skip_indices = indices[:2]

    # Generate all samples in one batch
    sampler = DeterministicUniformSampler(batch_size=2**n_indices)

    batches = sampler.generate_batches(indices)
    first_batch = list(next(batches))
    assert first_batch, "First batch should not be empty"

    # Skip indices for the next batch of all samples
    sampler.skip_indices = skip_indices

    next_batch = list(next(batches))

    effective_outer_indices = np.setdiff1d(indices, skip_indices)
    assert len(next_batch) == len(effective_outer_indices) * 2 ** (n_indices - 1)
    for sample in next_batch:
        assert sample.idx not in skip_indices, (
            f"Sample with skipped index {sample.idx} found"
        )

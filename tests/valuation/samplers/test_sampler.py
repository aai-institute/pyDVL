from __future__ import annotations

import math
from functools import reduce
from itertools import islice, takewhile
from typing import Any, Callable, Iterator, Type

import numpy as np
import pytest
from more_itertools import flatten
from numpy.typing import NDArray

from pydvl.utils.numeric import logcomb, powerset
from pydvl.utils.types import Seed
from pydvl.valuation import EvaluationStrategy, IndexSampler
from pydvl.valuation.samplers import (
    AntitheticOwenSampler,
    AntitheticPermutationSampler,
    AntitheticSampler,
    ConstantSampleSize,
    DeltaShapleyNCSGDConfig,
    DeterministicPermutationSampler,
    DeterministicUniformSampler,
    FiniteNoIndexIteration,
    FiniteSequentialIndexIteration,
    FiniteSequentialSizeIteration,
    GridOwenStrategy,
    HarmonicSampleSize,
    IndexIteration,
    LOOSampler,
    MSRSampler,
    NoIndexIteration,
    OwenSampler,
    PermutationSampler,
    PowerLawSampleSize,
    RandomIndexIteration,
    RandomSizeIteration,
    RoundRobinSizeIteration,
    SequentialIndexIteration,
    StochasticSampler,
    StratifiedPermutationSampler,
    StratifiedSampler,
    UniformOwenStrategy,
    UniformSampler,
)
from pydvl.valuation.samplers.ame import AMESampler
from pydvl.valuation.samplers.permutation import PermutationSamplerBase
from pydvl.valuation.types import IndexSetT, Sample, SampleGenerator
from pydvl.valuation.utility.base import UtilityBase

from .. import recursive_make


def deterministic_samplers():
    return [
        (DeterministicUniformSampler, {}),
        (DeterministicPermutationSampler, {}),
        (LOOSampler, {}),
    ]


def improper_samplers():
    return [
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
                    {"n_samples_outer": lambda n=200: n, "seed": lambda s: s},
                ),
                "index_iteration": NoIndexIteration,
            },
        ),
        (
            AntitheticOwenSampler,
            {
                "outer_sampling_strategy": (
                    UniformOwenStrategy,
                    {"n_samples_outer": lambda n=200: n, "seed": lambda s: s},
                ),
                "index_iteration": NoIndexIteration,
            },
        ),
        (
            OwenSampler,
            {
                "outer_sampling_strategy": (
                    UniformOwenStrategy,
                    {"n_samples_outer": lambda n=200: n, "seed": lambda s: s},
                ),
                "index_iteration": FiniteNoIndexIteration,
            },
        ),
        (
            AntitheticOwenSampler,
            {
                "outer_sampling_strategy": (
                    UniformOwenStrategy,
                    {"n_samples_outer": lambda n=200: n, "seed": lambda s: s},
                ),
                "index_iteration": FiniteNoIndexIteration,
            },
        ),
        (MSRSampler, {"seed": lambda seed: seed}),
    ]


def permutation_samplers():
    return [
        (PermutationSampler, {"seed": lambda seed: seed}),
        (AntitheticPermutationSampler, {"seed": lambda seed: seed}),
    ]


def amesampler_samplers():
    return [
        (
            AMESampler,
            {"index_iteration": RandomIndexIteration, "seed": lambda seed: seed},
        ),
        (
            AMESampler,
            {"index_iteration": SequentialIndexIteration, "seed": lambda seed: seed},
        ),
    ]



def powerset_samplers():
    ret = []

    for sampler in (UniformSampler, AntitheticSampler):
        for idx_it in (RandomIndexIteration, SequentialIndexIteration):
            ret.append(
                (sampler, {"index_iteration": idx_it, "seed": lambda seed: seed})
            )
    return ret


def stratified_samplers(n_samples_per_index: int = 32):
    # dummy_config = DeltaShapleyNCSGDConfig(
    #     lipschitz_grad=1,
    #     lipschitz_loss=1,
    #     lr_factor=1,
    #     n_sgd_iter=1,
    #     max_loss=1,
    #     n_val=10,
    #     n_train=10,
    #     eps=0.1,
    #     delta=0.1,
    #     version="theorem7",
    # )

    sample_size_strategies = [
        (
            ConstantSampleSize,
            {
                "n_samples": lambda n=n_samples_per_index: n,
                "lower_bound": lambda l=2: l,
                "upper_bound": lambda u=4: u,
            },
        ),
        (ConstantSampleSize, {"n_samples": lambda n=n_samples_per_index: n}),
        (
            HarmonicSampleSize,
            {
                "n_samples": lambda n=n_samples_per_index: n,
                "lower_bound": lambda l=1: l,
                "upper_bound": lambda u=None: u,
            },
        ),
        (
            PowerLawSampleSize,
            {"n_samples": lambda n=n_samples_per_index: n, "exponent": lambda e=0.5: e},
        ),
        # (DeltaShapleyNCSGDSampleSize, {"config": dummy_config}),
    ]

    sample_sizes_iterations = [
        FiniteSequentialSizeIteration,
        RandomSizeIteration,
        RoundRobinSizeIteration,
    ]

    index_iterations = [
        RandomIndexIteration,
        SequentialIndexIteration,
        FiniteSequentialIndexIteration,
    ]

    ret = []
    for ss in sample_size_strategies:
        ret.append(
            (
                StratifiedPermutationSampler,
                {"sample_sizes": ss, "seed": lambda seed: seed},
            )
        )
        for s_it in sample_sizes_iterations:
            for i_it in index_iterations:
                if (  # Finite iterations don't mix well with random ones
                    i_it == FiniteSequentialIndexIteration
                    and s_it != FiniteSequentialSizeIteration
                ):
                    continue

                ret.append(
                    (
                        StratifiedSampler,
                        {
                            "sample_sizes": ss,
                            "sample_sizes_iteration": s_it,
                            "index_iteration": i_it,
                            "seed": lambda seed: seed,
                        },
                    )
                )
    return ret


def owen_samplers():
    return [
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
    ]


def random_samplers(proper: bool = False, n_samples_per_index: int = 32):
    """Use this as parameter values in pytest.mark.parametrize for parameters
    "sampler_cls, sampler_kwargs"

    Build the objects with recursive_make(sampler_cls, sampler_kwargs, **lambda_args)
    where lambda args are named as the key in the dictionary that contains the lambda.
    """
    return (
        permutation_samplers()
        + powerset_samplers()
        + amesampler_samplers()
        + stratified_samplers(n_samples_per_index=n_samples_per_index)
        + owen_samplers()
        + (improper_samplers() if not proper else [])
    )


@pytest.mark.parametrize(
    "sampler_cls, sampler_kwargs, indices, max_samples, expected_batches",
    [
        (
            DeterministicUniformSampler,
            {"batch_size": 1},
            np.array([1, 2, 3]),
            5,
            [
                [Sample(1, np.array([]))],
                [Sample(1, np.array([2]))],
                [Sample(1, np.array([3]))],
                [Sample(1, np.array([2, 3]))],
                [Sample(2, np.array([]))],
            ],
        ),
        (
            DeterministicUniformSampler,
            {"index_iteration": FiniteSequentialIndexIteration, "batch_size": 4},
            np.array([1, 2, 3]),
            8,
            [
                [
                    Sample(1, np.array([])),
                    Sample(1, np.array([2])),
                    Sample(1, np.array([3])),
                    Sample(1, np.array([2, 3])),
                ],
                [
                    Sample(2, np.array([])),
                    Sample(2, np.array([1])),
                    Sample(2, np.array([3])),
                    Sample(2, np.array([1, 3])),
                ],
            ],
        ),
        (
            DeterministicUniformSampler,
            {"index_iteration": FiniteNoIndexIteration, "batch_size": 4},
            np.array([1, 2]),
            None,
            [
                [
                    Sample(None, np.array([])),
                    Sample(None, np.array([1])),
                    Sample(None, np.array([2])),
                    Sample(None, np.array([1, 2])),
                ],
            ],
        ),
        (
            DeterministicPermutationSampler,
            {},
            np.array([0, 1, 2]),
            None,
            [
                [Sample(None, np.array([0, 1, 2]))],
                [Sample(None, np.array([0, 2, 1]))],
                [Sample(None, np.array([1, 0, 2]))],
                [Sample(None, np.array([1, 2, 0]))],
                [Sample(None, np.array([2, 0, 1]))],
                [Sample(None, np.array([2, 1, 0]))],
            ],
        ),
        (
            LOOSampler,
            {},
            np.array([0, 1, 2]),
            None,
            [
                [Sample(0, np.array([1, 2]))],
                [Sample(1, np.array([0, 2]))],
                [Sample(2, np.array([0, 1]))],
            ],
        ),
    ],
    ids=[
        "deterministic_uniform_sampler_bs1",
        "deterministic_uniform_sampler_bs4",
        "deterministic_uniform_sampler_noindex",
        "deterministic_permutation_sampler",
        "loo_sampler",
    ],
)
def test_deterministic_samplers_batched(
    sampler_cls: Type[IndexSampler],
    sampler_kwargs: dict,
    indices: NDArray[np.int_],
    max_samples: int,
    expected_batches: list[list[Sample]],
):
    sampler = recursive_make(sampler_cls, sampler_kwargs)

    if max_samples is None:
        batches = list(sampler.generate_batches(indices))
    else:
        batches = list(
            takewhile(
                lambda _: sampler.n_samples <= max_samples,
                sampler.generate_batches(indices),
            )
        )

    assert len(batches) == len(expected_batches)

    for batch, expected_batch in zip(batches, expected_batches):
        batch = list(batch)
        assert len(batch) == len(expected_batch)
        for sample, expected in zip(batch, expected_batch):
            assert sample == expected


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
@pytest.mark.parametrize("batch_size, n_batches", [(1, 5), (2, 3)])
def test_sample_counter(
    sampler_cls, sampler_kwargs: dict, batch_size: int, n_batches: int, seed: int
):
    """Test that the sample counter indeed reflects the number of samples generated.

    This test was introduced after finding a bug in the DeterministicUniformSampler
    that was not caused by existing tests.
    """
    sampler = recursive_make(sampler_cls, sampler_kwargs, seed=seed)
    indices = np.arange(4)
    batches = list(islice(sampler.generate_batches(indices), n_batches))
    assert sampler.n_samples == len(list(flatten(batches)))


@pytest.mark.parametrize(
    "indices", [np.array([]), np.arange(3)], ids=["no_indices", "3_indices"]
)
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
    ids=[
        "DeterministicUniformSampler",
        "DeterministicUniformSampler-FiniteNoIndexIteration",
        "DeterministicPermutationSampler",
        "LOOSampler",
        "OwenSampler-Grid4-Inner2",
        "AntitheticOwenSampler-Grid4-Sequential",
        "StratifiedSampler-Harmonic32",
        "StratifiedSampler-PowerLaw13",
    ],
)
def test_length_for_finite_samplers(
    indices, sampler_cls, sampler_kwargs, expected_length
):
    sampler = recursive_make(sampler_cls, sampler_kwargs)
    assert sampler.sample_limit(indices) == expected_length(len(indices))
    assert len(list(sampler.generate_batches(indices))) == expected_length(len(indices))
    assert len(sampler) == expected_length(len(indices))


@pytest.mark.parametrize("sampler_cls, sampler_kwargs", random_samplers())
@pytest.mark.parametrize("indices", [np.empty(0), np.arange(3)])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_length_of_infinite_samplers(
    sampler_cls, sampler_kwargs, indices: NDArray, batch_size: int, seed: int
):
    sampler_kwargs |= {"batch_size": batch_size}
    sampler = recursive_make(sampler_cls, sampler_kwargs, seed=seed)
    if sampler.sample_limit(indices) is not None:
        pytest.skip(f"{sampler_cls.__name__} is a finite sampler")
    n_batches = 2 ** (len(indices) + 1)
    # check that we can generate samples that are longer than size of powerset
    batches = list(islice(sampler.generate_batches(indices), n_batches))
    if len(indices) > 0:
        assert sampler.sample_limit(indices) is None
        assert (
            len(list(flatten(batches))) == n_batches * batch_size == sampler.n_samples
        )
    else:
        assert sampler.sample_limit(indices) == 0

    with pytest.raises(TypeError):
        len(sampler)


def hash_indices(sampler_cls: Type[IndexSampler], indices: NDArray) -> int:
    """Hashes the indices to a single integer.

    This assumes that the indices are prime numbers!
    """
    if issubclass(
        sampler_cls,
        (
            PermutationSampler,
            AntitheticPermutationSampler,
            StratifiedPermutationSampler,
        ),
    ):
        # Order matters for permutations!
        p = 31  # A prime number as base for polynomial hash function
        return reduce(lambda h, y: h * p + y, indices, 0)
    else:
        return reduce(lambda x, y: x * y, indices, 1)


@pytest.mark.parametrize("sampler_cls, sampler_kwargs", random_samplers())
def test_proper_reproducible(sampler_cls, sampler_kwargs: dict, seed):
    """Test that the sampler is reproducible."""
    indices = np.array([1, 3, 5, 7, 11, 13, 17, 19])
    samples1 = _create_seeded_sample_iter(sampler_cls, sampler_kwargs, indices, seed)
    samples2 = _create_seeded_sample_iter(sampler_cls, sampler_kwargs, indices, seed)

    seq1 = [hash_indices(sampler_cls, s.subset) for s in samples1]
    seq2 = [hash_indices(sampler_cls, s.subset) for s in samples2]

    assert np.all(seq1 == seq2)


@pytest.mark.parametrize("sampler_cls, sampler_kwargs", random_samplers())
def test_proper_stochastic(sampler_cls, sampler_kwargs, seed, seed_alt):
    """Test that the sampler is stochastic."""
    indices = np.array([1, 3, 5, 7, 11, 13, 17, 19])
    samples1 = _create_seeded_sample_iter(sampler_cls, sampler_kwargs, indices, seed)
    samples2 = _create_seeded_sample_iter(
        sampler_cls, sampler_kwargs, indices, seed_alt
    )

    seq1 = [hash_indices(sampler_cls, s.subset) for s in samples1]
    seq2 = [hash_indices(sampler_cls, s.subset) for s in samples2]

    assert np.any(seq1 != seq2)


def _create_seeded_sample_iter(
    sampler_cls: Type[StochasticSampler],
    sampler_kwargs: dict[str, Any],
    indices: NDArray[np.int_],
    seed: Seed,
) -> Iterator:
    """Returns a flattened iterator over samples generated by the sampler."""
    sampler_kwargs["seed"] = seed
    sampler = recursive_make(
        sampler_cls,
        sampler_kwargs,
        seed=seed,
        # Since we sample very few sets, increase the likelihood of non-empty sets for Owen
        # (these parameters are passed to lambdas in the structure of the sampler_kwargs)
        n_samples_inner=1,
        n_samples_outer=4,
    )
    max_iterations = len(indices)
    sample_stream = map(
        lambda batch: list(batch)[0],
        islice(sampler.generate_batches(indices), max_iterations),
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

    # Sample and count the number of subsets of each size
    n = len(indices)
    n_batches = 2 ** (n * 2)  # go high to be sure

    sampler = recursive_make(sampler_cls, sampler_kwargs, seed=seed)

    # These samplers return samples up to the size of the whole index set
    if issubclass(sampler_cls, MSRSampler) or issubclass(
        getattr(sampler, "_index_iterator_cls", IndexIteration), NoIndexIteration
    ):
        effective_n = n
    else:
        effective_n = n - 1

    # HACK: MSR actually has same distribution as uniform over 2^(N-i)
    log_fudge = -math.log(2) if issubclass(sampler_cls, MSRSampler) else 0.0

    subset_len_probs = np.zeros(effective_n + 1)
    for batch in islice(sampler.generate_batches(indices), n_batches):
        for sample in batch:
            if issubclass(sampler_cls, StratifiedPermutationSampler):
                lb, ub = sample.lower_bound, sample.upper_bound
                subset_len_probs[lb : ub + 1] += 1
            elif issubclass(sampler_cls, PermutationSamplerBase):
                # The eval strategy iterates through the whole permutation, which is
                # effectively equivalent to yielding every subset size, for each sample.
                subset_len_probs += 1
            else:
                subset_len_probs[len(sample.subset)] += 1
    subset_len_probs /= subset_len_probs.sum()

    expected_log_subset_len_probs = np.full(effective_n + 1, -np.inf)
    for k in range(effective_n + 1):
        # log_weight = log probability of sampling
        # So: no. of sets of size k in the powerset, times. prob of sampling S|k
        expected_log_subset_len_probs[k] = (
            logcomb(effective_n, k) + sampler.log_weight(n, k) + log_fudge
        )

    np.testing.assert_allclose(
        subset_len_probs, np.exp(expected_log_subset_len_probs), atol=0.05
    )


@pytest.mark.parametrize(
    "sampler_cls, sampler_kwargs, n_batches",
    [
        (DeterministicUniformSampler, {}, lambda n: 2 ** (n - 1)),
        (UniformSampler, {}, lambda n: 2 ** (n - 1)),
        (AntitheticSampler, {}, lambda n: 2 ** (n - 1)),
        (LOOSampler, {}, lambda n: n),
        (PermutationSampler, {}, lambda n: math.factorial(n)),
        (AntitheticPermutationSampler, {}, lambda n: math.factorial(n)),
    ],
)
@pytest.mark.parametrize(
    "indices, skip, expected",
    [
        (np.arange(5), np.array([2, 4]), [0, 1, 3]),
        (np.arange(3), np.empty(0), np.arange(3)),
        (np.empty(0), np.arange(6), np.empty(0)),
    ],
)
def test_skip_indices(
    sampler_cls, sampler_kwargs, n_batches, indices, skip, expected, seed
):
    sampler_kwargs["batch_size"] = 2
    sampler = recursive_make(sampler_cls, sampler_kwargs, seed=seed)
    sampler.skip_indices = skip

    # Check that the outer iteration skips indices:
    if hasattr(sampler, "index_iterable"):
        outer_indices = list(islice(sampler.index_iterable(indices), len(indices)))
        assert set(outer_indices) == set(expected)

    # Check that the generated samples skip indices...
    batches = list(
        islice(sampler.generate_batches(indices), max(1, n_batches(len(indices))))
    )
    all_samples = list(flatten(batches))

    # ... in sample.subset for permutation samplers
    if isinstance(sampler, PermutationSamplerBase):
        assert all(
            all(idx in expected for idx in sample.subset) for sample in all_samples
        )
    else:  # ... in sample.idx for other samplers
        assert all(sample.idx in expected for sample in all_samples)


class TestBatchSampler(IndexSampler):
    def __init__(self, batch_size):
        super().__init__(batch_size)

    def sample_limit(self, indices: IndexSetT) -> int | None: ...

    def generate(self, indices: IndexSetT) -> SampleGenerator:
        yield from (Sample(idx, np.empty_like(indices)) for idx in indices)

    def log_weight(self, n: int, subset_len: int) -> float: ...

    def make_strategy(
        self,
        utility: UtilityBase,
        log_coefficient: Callable[[int, int], float] | None = None,
    ) -> EvaluationStrategy: ...


@pytest.mark.parametrize("indices", [np.arange(1), np.arange(23)])
@pytest.mark.parametrize("batch_size", [1, 2, 7])
def test_batching(indices, batch_size):
    sampler = TestBatchSampler(batch_size)
    batches = list(sampler.generate_batches(indices))

    assert all(hasattr(batch, "__iter__") for batch in batches)

    assert len(batches) == math.ceil(len(indices) / batch_size)
    assert len(batches[-1]) == len(indices) % batch_size or batch_size

    all_samples = list(flatten(batches))
    assert len(all_samples) == len(indices)

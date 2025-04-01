"""This module tests some of the components of stratified samplers,
but the main features are tested in test_sampler.py"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from pydvl.valuation import (
    ConstantSampleSize,
    FiniteSequentialSizeIteration,
    GroupTestingSampleSize,
    HarmonicSampleSize,
    PowerLawSampleSize,
    RandomSizeIteration,
    RoundRobinSizeIteration,
    SampleSizeStrategy,
)
from pydvl.valuation.samplers.stratified import LinearSampleSize
from tests.valuation import recursive_make


class MockSampleSizeStrategy(SampleSizeStrategy):
    def __init__(self, sample_sizes: list[int]):
        super().__init__(n_samples=sum(sample_sizes))
        self._sample_sizes = np.array(sample_sizes, dtype=int)

    def sample_sizes(self, n_indices: int, probs: bool = True) -> NDArray[np.int64]:
        if probs:
            return self._sample_sizes / np.sum(self._sample_sizes)
        return self._sample_sizes

    def fun(self, n_indices: int, subset_len: int) -> float:
        raise NotImplementedError("Shouldn't happen")


@pytest.mark.parametrize(
    "iteration_cls, sample_sizes, expected_output",
    [
        (RoundRobinSizeIteration, [], []),
        (RoundRobinSizeIteration, [2], [(0, 1), (0, 1)]),
        (RoundRobinSizeIteration, [0, 2], [(1, 1), (1, 1)]),
        (
            RoundRobinSizeIteration,
            [2, 3, 1],
            [(0, 1), (1, 1), (2, 1), (0, 1), (1, 1), (1, 1)],
        ),
        (FiniteSequentialSizeIteration, [], []),
        (FiniteSequentialSizeIteration, [2], [(0, 2)]),
        (FiniteSequentialSizeIteration, [0, 2], [(1, 2)]),
        (FiniteSequentialSizeIteration, [2, 3, 1], [(0, 2), (1, 3), (2, 1)]),
    ],
)
def test_deterministic_iterations(iteration_cls, sample_sizes, expected_output):
    n_indices = len(sample_sizes)
    strategy = MockSampleSizeStrategy(sample_sizes)
    iterable = iteration_cls(strategy, n_indices)
    output = list(iter(iterable))
    assert output == expected_output


@pytest.mark.parametrize("sample_sizes", [[1], [0, 1], [2, 3, 1]])
def test_random_iteration(sample_sizes, seed):
    n_indices = len(sample_sizes) - 1
    strategy = MockSampleSizeStrategy(sample_sizes)
    iterable = RandomSizeIteration(strategy, n_indices, seed)
    counts = np.zeros(n_indices + 1, dtype=float)
    sizes = strategy.sample_sizes(n_indices)
    for _ in range(100 * (n_indices + 1)):
        for size, n_samples in iterable:
            counts[size] += n_samples

    counts /= max(1, counts.sum())
    np.testing.assert_allclose(counts, sizes / sum(sizes), rtol=0.1, atol=0.1)


@pytest.mark.parametrize(
    "lower_bound, upper_bound, n_indices, subset_len, expected",
    [
        (0, None, 5, 0, 1 / 6),
        (0, None, 5, 5, 1 / 6),
        (1, 4, 5, 0, 0.0),
        (1, 4, 5, 1, 1 / 4),
        (1, 4, 5, 4, 1 / 4),
        (1, 4, 5, 5, 0.0),
        (None, 3, 5, 0, 1 / 4),
        (None, 3, 5, 3, 1 / 4),
        (None, 3, 5, 4, 0.0),
    ],
)
def test_constant_sample_size_fun(
    lower_bound, upper_bound, n_indices, subset_len, expected
):
    strategy = ConstantSampleSize(1, lower_bound, upper_bound)
    np.testing.assert_allclose(
        strategy.sample_sizes(n_indices, probs=True)[subset_len], expected
    )


@pytest.mark.parametrize(
    "strategy_cls, strategy_kwargs",
    [
        (LinearSampleSize, {"scale": 1}),
        (
            ConstantSampleSize,
            {"lower_bound": lambda n: n // 3, "upper_bound": lambda n: 2 * n // 3},
        ),
        (GroupTestingSampleSize, {}),
        (
            HarmonicSampleSize,
            {"lower_bound": lambda n: n // 3, "upper_bound": lambda n: 2 * n // 3},
        ),
        (
            PowerLawSampleSize,
            {
                "exponent": lambda e: e,
                "lower_bound": lambda n: n // 3,
                "upper_bound": lambda n: 2 * n // 3,
            },
        ),
    ],
)
@pytest.mark.parametrize("n_samples, n_indices", [(None, 5), (10, 7), (11, 10)])
@pytest.mark.parametrize("probs", [True, False])
@pytest.mark.parametrize("scale", [-1 / 3, -1.0, -np.pi])
def test_sample_sizes_sum(
    strategy_cls, strategy_kwargs, n_samples, n_indices, probs, scale
):
    strategy_kwargs["n_samples"] = n_samples
    strategy = recursive_make(
        strategy_cls,
        strategy_kwargs,
        lower_bound=n_indices,  # all these are fed to the respective lambdas above
        upper_bound=n_indices,
        exponent=scale,
        scale=scale,
    )
    sizes = strategy.sample_sizes(n_indices, probs=probs)
    if probs:
        np.testing.assert_allclose(sum(sizes), 1.0)
    else:
        assert np.all(np.floor(sizes) == sizes), "Quantized sizes must be integers"
        if n_samples is None:
            lb, ub = strategy.effective_bounds(n_indices)
            assert sum(sizes) >= ub - lb
        else:
            np.testing.assert_allclose(sum(sizes), n_samples)


@pytest.mark.parametrize("n_samples, n_indices", [(10, 7)])
@pytest.mark.parametrize("scale", [1 / 3, 1.0, np.pi])
def test_sample_sizes_quantization(n_samples, n_indices, scale):
    strategy = LinearSampleSize(scale=scale, offset=0, n_samples=n_samples)
    sizes = strategy.sample_sizes(n_indices, probs=False)
    assert np.all(np.floor(sizes) == sizes), "Quantized sizes must be integers"

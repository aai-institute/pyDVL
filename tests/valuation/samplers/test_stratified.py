"""This module tests some of the components of stratified samplers,
but the main features are tested in test_sampler.py"""

import numpy as np
import pytest
from numpy.typing import NDArray

from pydvl.valuation import (
    ConstantSampleSize,
    DeterministicSizeIteration,
    RoundRobinIteration,
    SampleSizeStrategy,
)


class MockSampleSizeStrategy(SampleSizeStrategy):
    def __init__(self, sample_sizes: list[int]):
        super().__init__(n_samples=sum(sample_sizes))
        self._sample_sizes = np.array(sample_sizes, dtype=int)

    def sample_sizes(self, n_indices: int, quantize: bool = True) -> NDArray[np.int_]:
        return self._sample_sizes

    def fun(self, n_indices: int, subset_len: int) -> float:
        raise NotImplementedError("Shouldn't happen")


@pytest.mark.parametrize(
    "sample_sizes, expected_output",
    [
        ([], []),
        ([1], [(0, 1)]),
        ([0, 1], [(1, 1)]),
        ([2, 3, 1], [(0, 1), (1, 1), (2, 1), (0, 1), (1, 1), (1, 1)]),
    ],
)
def test_round_robin_mode(sample_sizes, expected_output):
    n_indices = len(sample_sizes)
    strategy = MockSampleSizeStrategy(sample_sizes)
    round_robin_mode = RoundRobinIteration(strategy, n_indices)
    output = list(iter(round_robin_mode))
    assert output == expected_output


@pytest.mark.parametrize(
    "sample_sizes, expected_output",
    [
        ([], []),
        ([1], [(0, 1)]),
        ([0, 1], [(1, 1)]),
        ([2, 3, 1], [(0, 2), (1, 3), (2, 1)]),
    ],
)
def test_deterministic_mode(sample_sizes, expected_output):
    n_indices = len(sample_sizes)
    strategy = MockSampleSizeStrategy(sample_sizes)
    deterministic_mode = DeterministicSizeIteration(strategy, n_indices)
    output = list(iter(deterministic_mode))

    assert output == expected_output


@pytest.mark.parametrize(
    "lower_bound, upper_bound, n_indices, subset_len, expected",
    [
        (0, None, 5, 0, 1.0),
        (0, None, 5, 5, 1.0),
        (1, 4, 5, 0, 0.0),
        (1, 4, 5, 1, 1.0),
        (1, 4, 5, 4, 1.0),
        (1, 4, 5, 5, 0.0),
        (None, 3, 5, 0, 1.0),
        (None, 3, 5, 3, 1.0),
        (None, 3, 5, 4, 0.0),
    ],
)
def test_constant_sample_size_fun(
    lower_bound, upper_bound, n_indices, subset_len, expected
):
    strategy = ConstantSampleSize(1, lower_bound, upper_bound)
    assert strategy.fun(n_indices, subset_len) == expected


class LinearSampleSize(SampleSizeStrategy):
    def __init__(self, n_samples: int, scale: float):
        super().__init__(n_samples)
        self.scale = scale

    def fun(self, n_indices: int, subset_len: int) -> float:
        return int(subset_len * self.scale)


@pytest.mark.parametrize("n_samples, n_indices", [(1, 5), (10, 7)])
@pytest.mark.parametrize("quantize", [True, False])
@pytest.mark.parametrize("scale", [1 / 3, 1.0, np.pi])
def test_sample_sizes_sum(n_samples, n_indices, quantize, scale):
    strategy = LinearSampleSize(n_samples, scale)
    sizes = strategy.sample_sizes(n_indices, quantize=quantize)
    np.testing.assert_allclose(sum(sizes), n_samples)


@pytest.mark.parametrize("n_samples, n_indices", [(1, 5), (10, 7)])
@pytest.mark.parametrize("scale", [1 / 3, 1.0, np.pi])
def test_sample_sizes_quantization(n_samples, n_indices, scale):
    strategy = LinearSampleSize(n_samples, scale)
    sizes = strategy.sample_sizes(n_indices, quantize=True)
    assert np.all(np.floor(sizes) == sizes), "Quantized sizes must be integers"

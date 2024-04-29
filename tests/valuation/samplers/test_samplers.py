from itertools import takewhile

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pydvl.valuation.samplers.powerset import DeterministicUniformSampler


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


def _check_idxs(batches, expected):
    for batch, expected_batch in zip(batches, expected):
        for sample, expected_idx in zip(batch, expected_batch):
            assert sample.idx == expected_idx


def _check_subsets(batches, expected):
    for batch, expected_batch in zip(batches, expected):
        for sample, expected_subset in zip(batch, expected_batch):
            assert_array_equal(sample.subset, expected_subset)

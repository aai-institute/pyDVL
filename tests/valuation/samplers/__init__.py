from __future__ import annotations

from numpy.testing import assert_array_equal

from pydvl.valuation.types import ClasswiseSample

__all__ = ["_check_idxs", "_check_subsets", "_check_classwise_batches"]


def _check_idxs(batches, expected):
    for batch, expected_batch in zip(batches, expected):
        for sample, expected_idx in zip(batch, expected_batch):
            assert sample.idx == expected_idx


def _check_subsets(batches, expected):
    for batch, expected_batch in zip(batches, expected):
        for sample, expected_subset in zip(batch, expected_batch):
            assert_array_equal(sample.subset, expected_subset)


def _check_classwise_batches(
    batches: list[list[ClasswiseSample]], expected_batches: list[list[ClasswiseSample]]
) -> None:
    assert len(batches) == len(expected_batches), (
        f"{len(batches)=} != {len(expected_batches)=}"
    )
    for batch, expected_batch in zip(batches, expected_batches):
        for sample, expected_sample in zip(batch, expected_batch):
            assert sample == expected_sample

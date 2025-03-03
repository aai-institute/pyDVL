import numpy as np
import pytest

from pydvl.valuation import FiniteNoIndexIteration
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.samplers import (
    ClasswiseSampler,
    DeterministicPermutationSampler,
    DeterministicUniformSampler,
)
from pydvl.valuation.types import ClasswiseSample


@pytest.mark.parametrize(
    "data, expected_batches",
    [
        (
            Dataset(
                x=np.asarray([0.0, 0.5, 1.0]).reshape(-1, 1), y=np.asarray([0, 1, 1])
            ),
            [
                [
                    ClasswiseSample(
                        idx=None,
                        subset=np.asarray([0]),
                        label=0,
                        ooc_subset=np.asarray([1]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=None,
                        subset=np.asarray([1, 2]),
                        label=1,
                        ooc_subset=np.asarray([0]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=None,
                        subset=np.asarray([2, 1]),
                        label=1,
                        ooc_subset=np.asarray([0]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=None,
                        subset=np.asarray([0]),
                        label=0,
                        ooc_subset=np.asarray([2]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=None,
                        subset=np.asarray([0]),
                        label=0,
                        ooc_subset=np.asarray([1, 2]),
                    )
                ],
            ],
        ),
        (
            Dataset(
                x=np.asarray([0.0, 0.5, 1.0, 1.5]).reshape(-1, 1),
                y=np.asarray([0, 0, 1, 1]),
            ),
            [
                [
                    ClasswiseSample(
                        idx=None,
                        subset=np.asarray([0, 1]),
                        label=0,
                        ooc_subset=np.asarray([2]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=None,
                        subset=np.asarray([1, 0]),
                        label=0,
                        ooc_subset=np.asarray([2]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=None,
                        subset=np.asarray([2, 3]),
                        label=1,
                        ooc_subset=np.asarray([0]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=None,
                        subset=np.asarray([3, 2]),
                        label=1,
                        ooc_subset=np.asarray([0]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=None,
                        subset=np.asarray([0, 1]),
                        label=0,
                        ooc_subset=np.asarray([3]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=None,
                        subset=np.asarray([1, 0]),
                        label=0,
                        ooc_subset=np.asarray([3]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=None,
                        subset=np.asarray([2, 3]),
                        label=1,
                        ooc_subset=np.asarray([1]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=None,
                        subset=np.asarray([3, 2]),
                        label=1,
                        ooc_subset=np.asarray([1]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=None,
                        subset=np.asarray([0, 1]),
                        label=0,
                        ooc_subset=np.asarray([2, 3]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=None,
                        subset=np.asarray([1, 0]),
                        label=0,
                        ooc_subset=np.asarray([2, 3]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=None,
                        subset=np.asarray([2, 3]),
                        label=1,
                        ooc_subset=np.asarray([0, 1]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=None,
                        subset=np.asarray([3, 2]),
                        label=1,
                        ooc_subset=np.asarray([0, 1]),
                    )
                ],
            ],
        ),
    ],
)
def test_classwise_sampler(data, expected_batches):
    in_class_sampler = DeterministicPermutationSampler()
    out_of_class_sampler = DeterministicUniformSampler(
        index_iteration=FiniteNoIndexIteration
    )
    sampler = ClasswiseSampler(
        in_class=in_class_sampler, out_of_class=out_of_class_sampler
    )

    batches = list(sampler.from_data(data))
    assert len(batches) == len(expected_batches), (
        f"{len(batches)=} != {len(expected_batches)=}"
    )
    for batch, expected_batch in zip(batches, expected_batches):
        for sample, expected_sample in zip(batch, expected_batch):
            assert sample == expected_sample

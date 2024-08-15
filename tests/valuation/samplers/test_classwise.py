import numpy as np
import pytest

from pydvl.valuation.dataset import Dataset
from pydvl.valuation.samplers import (
    ClasswiseSampler,
    DeterministicPermutationSampler,
    DeterministicUniformSampler,
)
from pydvl.valuation.types import ClasswiseSample

from . import _check_classwise_batches


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
                        idx=-1,
                        subset=np.asarray([0]),
                        label=0,
                        ooc_subset=np.asarray([1]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=-1,
                        subset=np.asarray([1, 2]),
                        label=1,
                        ooc_subset=np.asarray([0]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=-1,
                        subset=np.asarray([2, 1]),
                        label=1,
                        ooc_subset=np.asarray([0]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=-1,
                        subset=np.asarray([0]),
                        label=0,
                        ooc_subset=np.asarray([2]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=-1,
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
                        idx=-1,
                        subset=np.asarray([0, 1]),
                        label=0,
                        ooc_subset=np.asarray([2]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=-1,
                        subset=np.asarray([1, 0]),
                        label=0,
                        ooc_subset=np.asarray([2]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=-1,
                        subset=np.asarray([2, 3]),
                        label=1,
                        ooc_subset=np.asarray([0]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=-1,
                        subset=np.asarray([3, 2]),
                        label=1,
                        ooc_subset=np.asarray([0]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=-1,
                        subset=np.asarray([0, 1]),
                        label=0,
                        ooc_subset=np.asarray([3]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=-1,
                        subset=np.asarray([1, 0]),
                        label=0,
                        ooc_subset=np.asarray([3]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=-1,
                        subset=np.asarray([2, 3]),
                        label=1,
                        ooc_subset=np.asarray([1]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=-1,
                        subset=np.asarray([3, 2]),
                        label=1,
                        ooc_subset=np.asarray([1]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=-1,
                        subset=np.asarray([0, 1]),
                        label=0,
                        ooc_subset=np.asarray([2, 3]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=-1,
                        subset=np.asarray([1, 0]),
                        label=0,
                        ooc_subset=np.asarray([2, 3]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=-1,
                        subset=np.asarray([2, 3]),
                        label=1,
                        ooc_subset=np.asarray([0, 1]),
                    )
                ],
                [
                    ClasswiseSample(
                        idx=-1,
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
    out_of_class_sampler = DeterministicUniformSampler()
    sampler = ClasswiseSampler(
        in_class=in_class_sampler, out_of_class=out_of_class_sampler
    )
    batches = list(sampler.from_data(data))
    _check_classwise_batches(batches, expected_batches)

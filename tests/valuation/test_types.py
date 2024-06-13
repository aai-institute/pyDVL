import numpy as np
from numpy.testing import assert_array_equal

from pydvl.valuation.types import Sample


def test_sample():
    sample = Sample(idx=1, subset=np.array([2, 3, 4]))
    new_sample = sample.with_idx_in_subset()
    assert new_sample.idx == 1
    assert_array_equal(new_sample.subset, np.array([2, 3, 4, 1]))

import numpy as np
import pytest

from valuation.utils import vanishing_derivatives
from valuation.utils.numeric import powerset, random_powerset, \
    random_subset_indices


def test_dataset_len(boston_dataset):
    assert len(boston_dataset) == len(boston_dataset.x_train) == 404
    assert len(boston_dataset.x_train) + len(boston_dataset.x_test) == 506


def test_vanishing_derivatives():
    # 1/x for x>1e3
    vv = 1 / np.arange(1000, 1100, step=1).reshape(10, -1)
    assert vanishing_derivatives(vv, 7, 1e-2) == 10


def test_powerset():
    assert set(powerset((1, 2))) == {(), (1,), (1, 2), (2,)}
    assert set(powerset([])) == {()}
    with pytest.raises(TypeError):
        set(powerset(1))


@pytest.mark.timeout(3)
@pytest.mark.parametrize("n", [0, 1, 14])
def test_random_subset_indices(n):
    # TODO: compute lower bound for number of samples.
    m = 2**n
    indices = []
    for c in range(m):
        indices.extend(random_subset_indices(n))
    if n == 0:
        assert indices == []
    elif n == 1:
        assert np.alltrue([s in [0] for s in indices])
    else:
        frequencies, _ = np.histogram(indices, range(n+1), density=True)
        # FIXME: 10% relative error sucks, be more precise
        assert np.allclose(frequencies, np.mean(frequencies), rtol=1e-1)


@pytest.mark.parametrize("n", [0, 8])
def test_random_powerset(n):
    indices = np.arange(n)
    m = 2 ** (len(indices) + 2)  # Just to be safe...
    sets = []
    for s in random_powerset(indices, max_subsets=m):
        sets.append(tuple(s))
    missing = []
    for s in powerset(indices):
        s = tuple(s)
        try:
            sets.remove(s)
        except ValueError:
            missing.append(s)

    # FIXME: test convergence in expectation to 0 as m->\ifty
    assert len(missing) <= np.ceil(0.05*2**n)

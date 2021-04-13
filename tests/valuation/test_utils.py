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
    with pytest.raises(TypeError):
        set(powerset(1))

    assert set(powerset([])) == {()}
    assert set(powerset((1, 2))) == {(), (1,), (1, 2), (2,)}

    # Check correct number of sets of each size
    n = 10
    sizes = np.zeros(n+1)
    for s in powerset(range(n)):
        sizes[len(s)] += 1

    assert all([np.math.comb(n, j) for j in range(n+1)] == sizes)


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
        assert np.all([s in [0] for s in indices])
    else:
        frequencies, _ = np.histogram(indices, range(n+1), density=True)
        # FIXME: 10% relative error sucks, be more precise
        assert np.allclose(frequencies, np.mean(frequencies), rtol=1e-1)


@pytest.mark.timeout(5)
@pytest.mark.parametrize("n", [0, 8])
def test_random_powerset(n):
    with pytest.raises(TypeError):
        set(random_powerset(1, max_subsets=1))

    indices = np.arange(n)
    # TODO: compute eps,delta bound for sample complexity
    m = 2 ** int(len(indices)*2)
    sets = set()
    sizes = np.zeros(n + 1)
    for s in random_powerset(indices, max_subsets=m):
        sets.add(tuple(s))
        sizes[len(s)] += 1

    missing = set()
    for s in map(tuple, powerset(indices)):
        try:
            sets.remove(s)
        except KeyError:
            missing.add(s)

    # FIXME: non deterministic
    # FIXME: test convergence in expectation to 0 as m->\ifty
    assert len(missing) <= np.ceil(0.01*2**n)

    # Check distribution of set sizes
    sizes /= sum(sizes)
    exact_sizes = np.array([np.math.comb(n, j) for j in range(n+1)]) / 2**n
    assert np.allclose(sizes, exact_sizes, rtol=0.1)

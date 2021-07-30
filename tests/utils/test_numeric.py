import numpy as np
import pytest

from valuation.utils.numeric import powerset, random_powerset, \
    spearman, vanishing_derivatives


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


@pytest.mark.parametrize(
    "x, y, expected",
    [([], [], ValueError),
     ([1], [1], TypeError),
     ([1, 2, 3], [1, 2, 3], 1.0),
     ([1, 2, 3], [3, 2, 1], -1.0),
     (np.arange(1, 4), np.arange(4, 7), ValueError),
     # FIXME: non deterministic test
     (np.random.permutation(np.arange(100)),
      np.random.permutation(np.arange(100)), (0.0, 0.1))])
def test_spearman(x, y, expected):
    if isinstance(expected, float):
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        assert spearman(x, y) == expected
    elif isinstance(expected, tuple):
        value, atol = expected
        assert np.isclose(spearman(x, y), value, atol=atol)
    else:
        with pytest.raises(expected):
            spearman(x, y)

import numpy as np
import pytest

from valuation.utils import MemcachedConfig, available_cpus
from valuation.utils.numeric import (
    powerset,
    random_powerset,
    spearman,
    vanishing_derivatives,
)


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
    sizes = np.zeros(n + 1)
    for s in powerset(range(n)):
        sizes[len(s)] += 1

    assert all([np.math.comb(n, j) for j in range(n + 1)] == sizes)


@pytest.mark.parametrize("n, max_subsets", [(0, 10), (1, 1e3), (4, 1e4)])
def test_random_powerset(n, max_subsets, memcache_client_config):
    """
    Tests that random powerset samples the same items as the powerset method, and with
    the same frequency.
    This is done sampling a sufficiently high number of times (max_subsets) and checking that all
    the powersets are sampled with the same frequency (max minus min number of samples must be
    much smaller than average sampling number.
    """
    s = np.arange(1, n + 1)
    num_cpus = available_cpus()
    result = random_powerset(
        s,
        max_subsets=max_subsets,
        num_jobs=num_cpus,
        client_config=memcache_client_config,
    )
    result_exact = set(powerset(s))
    count_powerset = {key: 0 for key in result_exact}

    for res_pow in result:
        res_pow = tuple(np.sort(res_pow))
        count_powerset[tuple(res_pow)] += 1
    value_counts = list(count_powerset.values())
    assert 5 * (np.max(value_counts) - np.min(value_counts)) < np.mean(value_counts)


@pytest.mark.parametrize(
    "x, y, expected",
    [
        ([], [], ValueError),
        ([1], [1], TypeError),
        ([1, 2, 3], [1, 2, 3], 1.0),
        ([1, 2, 3], [3, 2, 1], -1.0),
        (np.arange(1, 4), np.arange(4, 7), ValueError),
        # FIXME: non deterministic test
        pytest.param(
            np.random.permutation(np.arange(100)),
            np.random.permutation(np.arange(100)),
            (0.0, 0.1),
            marks=pytest.mark.skip("This test case is flaky."),
        ),
    ],
)
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

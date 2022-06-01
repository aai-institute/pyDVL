from functools import partial, reduce
from typing import List, Optional

import numpy as np
import pytest

from valuation.utils import MapReduceJob, available_cpus, map_reduce
from valuation.utils.caching import ClientConfig
from valuation.utils.numeric import (
    PowerSetDistribution,
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


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "n",
    [
        0,
        pytest.param(
            8, marks=pytest.mark.skip("This test case is flaky. Needs investigating")
        ),
    ],
)
def test_random_powerset(n, memcache_client_config):
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        set(random_powerset(1, max_subsets=1))

    delta = 0.2
    reps = 5
    job = MapReduceJob.from_fun(
        partial(_random_powerset, memcached_client_config=memcache_client_config)
    )
    results = map_reduce(job, [n], num_runs=reps)

    from tests.conftest import TolerateErrors

    delta_errors = TolerateErrors(int(delta * reps))
    for r in results:
        with delta_errors:
            assert np.all(r), results


def _random_powerset(
    data: List[int], *, memcached_client_config: Optional[ClientConfig] = None
):
    n = data[0]  # yuk... map_reduce always sends lists
    indices = np.arange(n)
    eps = 0.01
    # TODO: compute (ε,δ) bound for sample complexity
    m = 2 ** (int(n * 1.5))
    sets = set()
    sizes = np.zeros(n + 1)
    num_cpus = available_cpus() if n > 0 else 1

    def sampler(indices: np.ndarray) -> List[frozenset]:
        ss: List[frozenset] = []
        for s in random_powerset(
            indices,
            dist=PowerSetDistribution.WEIGHTED,
            max_subsets=1 + m // num_cpus,
            client_config=memcached_client_config,
        ):
            ss.append(frozenset(s))
        return ss

    def reducer(results: List[List[frozenset]]) -> List[frozenset]:
        return reduce(lambda x, y: x.append(y) or x, results[0], [])

    job = MapReduceJob.from_fun(sampler, reducer)
    runs = map_reduce(job, indices, num_jobs=num_cpus, num_runs=num_cpus)
    for result in runs:
        for s in result:
            sets.add(s)
            sizes[len(s)] += 1

    missing = set()
    for s in map(frozenset, powerset(indices)):
        try:
            sets.remove(s)
        except KeyError:
            missing.add(s)
    if len(missing) / 2**n > eps:
        return False

    # Check expected set sizes:
    # E[|S|] ≈ Empirical[|S|], or P(|S| = k) ≈ Freq(|S| = k)
    sizes /= sum(sizes)
    exact_sizes = np.array([np.math.comb(n, j) for j in range(n + 1)]) / 2**n
    if not np.allclose(sizes, exact_sizes, atol=eps):
        return False

    return True


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

import time

import numpy as np

from pydvl.value import ValuationResult
from pydvl.value.stopping import (
    StoppingCriterion,
    MaxChecks,
    MaxTime,
    MaxUpdates,
    MinUpdates,
    StandardError,
    HistoryDeviation,
)


def test_max_checks():
    """Test the MaxChecks stopping criterion."""
    v = ValuationResult.from_random(size=5)

    done = MaxChecks(None)
    for _ in range(10):
        assert not done(v)

    done = MaxChecks(5)
    for _ in range(4):
        assert not done(v)
    assert done(v)


def test_max_time():
    """Test the MaxTime stopping criterion."""
    v = ValuationResult.from_random(size=5)

    # No time limit.
    done = MaxTime(None)
    time.sleep(0.1)
    assert not done(v)

    # Time limit.
    done = MaxTime(0.5)
    assert not done(v)
    time.sleep(0.5)
    assert done(v)


def test_max_updates():
    """Test the MaxUpdates stopping criterion."""
    done = MaxUpdates(5)
    v = ValuationResult.from_random(size=7)
    assert not done(v)
    for _ in range(4):
        v += v
    assert done(v)


def test_min_updates():
    """Test the MinUpdates stopping criterion."""
    done = MinUpdates(5)
    v = ValuationResult.from_random(size=7)

    # Updating one index above the threshold should not trigger the criterion.
    for _ in range(6):
        v.update(0, 1)
    assert not done(v)

    # Instead all need to be updated the given number of times.
    for _ in range(4):
        v += v
    assert done(v)


def test_standard_error():
    """Test the StandardError stopping criterion."""
    eps = 0.1
    n = 5

    done = StandardError(threshold=eps)

    # Trivial case: no variance.
    v = ValuationResult(values=np.ones(n), variances=np.zeros(n))
    assert done(v)

    # Reduce the variance until the criterion is triggered.
    v = ValuationResult(values=np.ones(n), variances=np.ones(n))
    assert not done(v)

    # One value is being left out
    for _ in range(10):
        for idx in range(1, n):
            v.update(idx, 1)
    assert not done(v)

    # Update the final value
    for _ in range(10):
        v.update(0, 1)
    assert done(v)

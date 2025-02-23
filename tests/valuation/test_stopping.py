import math
from time import sleep

import numpy as np
import pytest

from pydvl.utils import Status
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.stopping import (
    AbsoluteStandardError,
    HistoryDeviation,
    MaxChecks,
    MaxTime,
    MaxUpdates,
    MinUpdates,
    RankCorrelation,
    StoppingCriterion,
    _make_criterion,
)


def test_stopping_criterion():
    with pytest.raises(TypeError):
        StoppingCriterion()

    StoppingCriterion.__abstractmethods__ = frozenset()
    done = StoppingCriterion()
    assert str(done) == "StoppingCriterion"
    assert done.modify_result is True


def test_stopping_criterion_composition():
    StoppingCriterion.__abstractmethods__ = frozenset()

    c = Status.Converged
    p = Status.Pending
    f = Status.Failed

    class AlwaysConverged(StoppingCriterion):
        def _check(self, result: ValuationResult) -> Status:
            self._converged = np.full_like(result.values, True, dtype=bool)
            return c

    class AlwaysPending(StoppingCriterion):
        def _check(self, result: ValuationResult) -> Status:
            self._converged = np.full_like(result.values, False, dtype=bool)
            return p

    class AlwaysFailed(StoppingCriterion):
        def _check(self, result: ValuationResult) -> Status:
            self._converged = np.full_like(result.values, False, dtype=bool)
            return f

    v = ValuationResult.from_random(5)

    nac = ~AlwaysConverged()
    nap = ~AlwaysPending()
    naf = ~AlwaysFailed()

    assert nac(v) == f
    assert nap(v) == c
    assert naf(v) == c
    assert not nac.converged.all()
    assert nap.converged.all()
    assert naf.converged.all()

    ac_and_ac = AlwaysConverged() & AlwaysConverged()
    ap_or_ap = AlwaysPending() | AlwaysPending()

    assert ac_and_ac(v) == c
    assert ap_or_ap(v) == p
    assert ac_and_ac.converged.all()
    assert not ap_or_ap.converged.all()

    ac_and_ap = AlwaysConverged() & AlwaysPending()
    ac_or_ap = AlwaysConverged() | AlwaysPending()

    assert ac_and_ap(v) == (c & p)
    assert ac_or_ap(v) == (c | p)
    assert not ac_and_ap.converged.all()
    assert ac_or_ap.converged.all()

    ac_and_ac_and_ac = AlwaysConverged() & AlwaysConverged() & AlwaysConverged()
    ap_and_ap_or_ap = AlwaysPending() & AlwaysPending() | AlwaysPending()

    assert ac_and_ac_and_ac(v) == c
    assert ap_and_ap_or_ap(v) == p
    assert ac_and_ac_and_ac.converged.all()
    assert not ap_and_ap_or_ap.converged.all()

    assert str(ac_and_ap) == "AlwaysConverged AND AlwaysPending"
    assert str(ac_or_ap) == "AlwaysConverged OR AlwaysPending"
    assert str(nac) == "NOT AlwaysConverged"
    assert str(nap) == "NOT AlwaysPending"


def test_count_update_composite_criteria():
    """Test that the field _count in sub-criteria is updated correctly for composite
    criteria."""

    class P(StoppingCriterion):
        def _check(self, result: ValuationResult) -> Status:
            return Status.Pending

    c1 = P()
    c2 = P()

    c = c1 & c2
    assert c._count == 0
    assert c(ValuationResult.empty()) == Status.Pending
    assert c._count == 1
    assert c1._count == c2._count == 1

    c = c1 | c2
    assert c._count == 0
    assert c(ValuationResult.empty()) == Status.Pending
    assert c._count == 1
    assert c1._count == c2._count == 2

    c = ~c1
    assert c._count == 0
    assert c(ValuationResult.empty()) == Status.Converged
    assert c._count == 1
    assert c1._count == 3


def test_minmax_updates():
    maxstop = MaxUpdates(10)
    assert str(maxstop) == "MaxUpdates(n_updates=10)"
    v = ValuationResult.from_random(5)
    v._counts = np.zeros(5, dtype=int)
    assert maxstop(v) == Status.Pending
    v._counts += np.ones(5, dtype=int) * 9
    assert maxstop(v) == Status.Pending
    v._counts[0] += 1
    assert maxstop(v) == Status.Converged
    assert maxstop.completion() == 1.0
    assert np.sum(maxstop.converged) >= 1
    maxstop.reset()
    assert maxstop.completion() == 0.0
    assert not maxstop.converged.any()

    minstop = MinUpdates(10)
    assert str(minstop) == "MinUpdates(n_updates=10)"
    v._counts = np.zeros(5, dtype=int)
    assert minstop(v) == Status.Pending
    v._counts += np.ones(5, dtype=int) * 9
    assert minstop(v) == Status.Pending
    v._counts[0] += 1
    assert minstop(v) == Status.Pending
    v._counts += np.ones(5, dtype=int)
    assert minstop(v) == Status.Converged
    assert minstop.completion() == 1.0
    assert minstop.converged.all()
    minstop.reset()
    assert minstop.completion() == 0.0
    assert not minstop.converged.any()


@pytest.mark.flaky(reruns=1)  # Allow for some flakiness due to timing
def test_max_time():
    v = ValuationResult.from_random(5)
    done = MaxTime(0.3)
    assert done(v) == Status.Pending
    sleep(0.3)
    assert done(v) == Status.Converged
    assert done.completion() == 1.0
    assert done.converged.all()
    done.reset()
    np.testing.assert_allclose(done.completion(), 0, atol=0.01)
    assert not done.converged.any()


@pytest.mark.parametrize("n_steps", [1, 42, 100])
@pytest.mark.parametrize("rtol", [0.01, 0.05])
def test_history_deviation(n_steps, rtol):
    """Values are equal and set to 1/t. The criterion will be fulfilled after
    t > (1+1/rtol) * n_steps iterations.
    """
    n = 5
    done = HistoryDeviation(n_steps=n_steps, rtol=rtol)
    threshold = math.ceil((1 + 1 / rtol) * n_steps)
    for t in range(1, threshold):
        v = ValuationResult(values=np.ones(n) / t, counts=np.full(n, t))
        assert done(v) == Status.Pending

    # FIXME: rounding errors mean that the threshold is not exactly as computed,
    #  but might be off by 1, so we check a couple of iterations to be sure that
    #  this works for any choice of n_steps and rtol
    status = Status.Pending
    for t in range(threshold, threshold + 2):
        v = ValuationResult(values=np.ones(n) / t, counts=np.full(n, t))
        status |= done(v)

    assert status == Status.Converged
    assert done.completion() == 1.0
    assert done.converged.all()
    done.reset()
    assert done.completion() == 0.0
    assert not done.converged.any()


def test_standard_error():
    """Test the AbsoluteStandardError stopping criterion."""
    eps = 0.1
    n = 5

    done = AbsoluteStandardError(threshold=eps, fraction=1.0, burn_in=0)

    # Trivial case: no variance.
    v = ValuationResult(values=np.ones(n), variances=np.zeros(n))
    assert done(v)
    assert done.completion() == 1.0
    assert done.converged.all()

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
    assert done.completion() == 1.0
    assert done.converged.all()
    done.reset()
    assert done.completion() == 0.0
    assert not done.converged.any()


def test_max_checks():
    """Test the MaxChecks stopping criterion."""
    v = ValuationResult.from_random(size=5)

    done = MaxChecks(None)
    for _ in range(10):
        assert not done(v)
    assert done.completion() == 0.0

    done = MaxChecks(5)
    for _ in range(4):
        assert not done(v)
    assert done(v)

    assert done.completion() == 1.0
    assert done.converged.all()
    done.reset()
    assert done.completion() == 0.0


def test_rank_correlation():
    """Test the RankCorrelation stopping criterion."""
    v = ValuationResult.zeros(indices=range(5))
    arr = np.arange(5)

    done = RankCorrelation(rtol=0.1, burn_in=10)
    for i in range(20):
        arr = np.roll(arr, 1)
        for j in range(5):
            v.update(j, arr[j] + 0.01 * j)
        assert not done(v)
    assert not done(v)
    assert done(v)

    done = RankCorrelation(rtol=0.1, burn_in=3)
    v = ValuationResult.from_random(size=5)
    assert not done(v)
    assert not done(v)
    assert not done(v)
    assert done(v)

    done = RankCorrelation(rtol=0.1, burn_in=2)
    v = ValuationResult.from_random(size=5)
    assert not done(v)
    assert not done(v)
    assert done(v)

    assert done.completion() == 1.0
    assert done.converged.all()
    done.reset()
    assert done.completion() == 0.0
    assert not done.converged.any()


@pytest.mark.parametrize(
    "criterion",
    [
        AbsoluteStandardError(0.1),
        HistoryDeviation(10, 0.1),
        MaxChecks(5),
        MaxTime(0.1),
        MaxUpdates(10),
        MinUpdates(10),
        RankCorrelation(0.1, 1),
    ],
)
def test_count(criterion):
    """Test that the _count attribute and count property of stoppingcriteria are properly updated"""
    assert criterion.count == 0
    criterion(ValuationResult.empty())
    assert criterion.count == 1
    criterion(ValuationResult.empty())
    assert criterion.count == 2
    criterion.reset()
    assert criterion.count == 0
    criterion(ValuationResult.empty())
    assert criterion.count == 1
    criterion(ValuationResult.empty())
    assert criterion.count == 2


# Test that the _memory attribute and memory property of stoppingcriteria are properly updated:
@pytest.mark.parametrize(
    "criterion",
    [
        HistoryDeviation(6, 0.1),
        RankCorrelation(0.1, 0),
    ],
)
def test_memory(criterion):
    r1 = ValuationResult.from_random(5)
    r2 = ValuationResult.from_random(5)

    assert np.all(criterion.memory.data == [])
    criterion(r1)
    np.testing.assert_equal(criterion.memory.data[-1], r1.values)
    np.testing.assert_equal(criterion.memory[-1], r1.values)

    criterion(r2)
    tmp = np.vstack((r1.values, r2.values))
    np.testing.assert_equal(criterion.memory.data[-2:], tmp)
    np.testing.assert_equal(criterion.memory[-2:], tmp)

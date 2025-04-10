import math
from itertools import islice
from time import sleep

import numpy as np
import pytest

from pydvl.utils import Status
from pydvl.valuation import IndexSampler
from pydvl.valuation.result import LogResultUpdater, ValuationResult
from pydvl.valuation.stopping import (
    AbsoluteStandardError,
    History,
    HistoryDeviation,
    MaxChecks,
    MaxSamples,
    MaxTime,
    MaxUpdates,
    MinUpdates,
    NoStopping,
    RankCorrelation,
    RollingMemory,
    StoppingCriterion,
)
from pydvl.valuation.types import ValueUpdate


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
            # self._converged is set to False by __call__
            return p

    class AlwaysFailed(StoppingCriterion):
        def _check(self, result: ValuationResult) -> Status:
            # self._converged is set to False by __call__
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

    ac_and_ac_and_ac = AlwaysConverged() & (AlwaysConverged() & AlwaysConverged())
    ap_and_ap_or_ap = AlwaysPending() & (AlwaysPending() | AlwaysPending())

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

    r = ValuationResult.from_random(5)

    c1 = P()
    c2 = P()

    c = c1 & c2
    assert c._count == 0
    assert c(r) == Status.Pending
    assert c._count == 1
    assert c1._count == c2._count == 1

    c = c1 | c2
    assert c._count == 0
    assert c(r) == Status.Pending
    assert c._count == 1
    assert c1._count == c2._count == 2

    c = ~c1
    assert c._count == 0
    assert c(r) == Status.Converged
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
    assert str(done) == "MaxTime(seconds=0.3)"


@pytest.mark.parametrize("n_steps", [1, 42])
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

    with pytest.raises(ValueError, match="rtol"):
        HistoryDeviation(n_steps=n_steps, rtol=-0.1)

    with pytest.raises(ValueError, match="rtol"):
        HistoryDeviation(n_steps=n_steps, rtol=1.1)


def test_standard_error():
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
    updater = LogResultUpdater(v)
    assert not done(v)

    # One value is being left out
    for _ in range(10):
        for idx in range(1, n):
            updater.process(ValueUpdate(idx, np.log(1.0), 1))
    assert not done(v)

    # Update the final value
    for _ in range(10):
        updater.process(ValueUpdate(0, np.log(1.0), 1))
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
    assert done.completion() == 0.0
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
    n = 5
    burn_factor = 4
    v = ValuationResult.zeros(indices=range(n))
    updater = LogResultUpdater(v)
    arr = np.arange(n)

    def update_all():
        for j in range(n):
            updater.process(ValueUpdate(j, np.log(arr[j] + 0.01 * j), 1))

    done = RankCorrelation(rtol=0.1, burn_in=n * burn_factor, fraction=1)
    for i in range(n * burn_factor):
        arr = np.roll(arr, 1)
        update_all()
        assert not done(v)

    # After reaching burn_in, the first correlation computation is triggered:
    update_all()
    assert not done(v)

    # The next correlation computation will trigger the criterion
    update_all()
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
        RankCorrelation(0.1, 1, 0.1),
    ],
)
def test_count(criterion):
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


def test_memory():
    r1 = np.arange(5)
    r2 = np.arange(5, 10)
    r3 = np.arange(10, 15)

    memory = RollingMemory(size=4, default=np.nan, dtype=np.float64)
    assert len(memory) == 0

    assert np.all(memory.data == [])
    memory.update(r1)
    assert len(memory) == 1
    np.testing.assert_equal(memory.data[-1], r1)
    np.testing.assert_equal(memory[-1], r1)

    memory.update(r2)
    assert len(memory) == 2
    tmp = np.vstack((r1, r2))
    np.testing.assert_equal(memory.data[-2:], tmp)
    np.testing.assert_equal(memory[-2:], tmp)

    memory.update(r3)
    assert len(memory) == 3
    tmp2 = np.vstack((r2, r3))
    np.testing.assert_equal(memory.data[-2:], tmp2)
    np.testing.assert_equal(memory[-2:], tmp2)
    np.testing.assert_equal(memory.data[-3:-1], tmp)
    np.testing.assert_equal(memory[-3:-1], tmp)
    np.testing.assert_equal(memory.data[-1:-3:-1], tmp2[::-1])
    np.testing.assert_equal(memory[-1:-3:-1], tmp2[::-1])
    np.testing.assert_equal(memory[[-1, -2]], tmp2[::-1])
    np.testing.assert_equal(memory[[-2, -1]], tmp2)

    mask = np.array([False, True, False, True])
    np.testing.assert_array_equal(memory[mask], np.vstack((r1, r3)))

    with pytest.raises(IndexError):
        memory[0]  # noqa
    with pytest.raises(IndexError):
        memory[-4]  # noqa
    with pytest.raises(IndexError):
        memory[[-1, -6]]  # noqa
    with pytest.raises(IndexError):
        memory[[-1, 0]]  # noqa
    with pytest.raises(TypeError):
        memory["invalid"]  # noqa
    with pytest.raises(TypeError):
        memory[object()]  # noqa
    with pytest.raises(TypeError):
        memory[["string"]]  # noqa


def test_no_stopping_without_sampler():
    result = ValuationResult.from_random(5)
    no_stop = NoStopping()
    status = no_stop(result)
    assert status == Status.Pending
    assert no_stop.completion() == 0.0
    np.testing.assert_equal(no_stop.converged, False)
    assert str(no_stop) == "NoStopping()"


def test_history():
    n_steps = 4
    size = 5
    history = History(n_steps=n_steps)
    for i in range(1, n_steps + 1):
        result = ValuationResult.from_random(size=size)
        status = history(result)
        assert status == Status.Pending
        assert history.completion() == 0.0
        np.testing.assert_equal(history.converged, False)
        assert all(history[-1] == result.values)
        assert len(history) == i
    assert history.count == n_steps
    assert history.data.shape == (n_steps, size)


class DummyFiniteSampler(IndexSampler):
    def __init__(self, total_samples: int = 10, batch_size: int = 1):
        super().__init__(batch_size=batch_size)
        self.total_samples = total_samples

    def sample_limit(self, indices):
        return self.total_samples

    def generate(self, indices):
        for i in range(self.total_samples):
            yield i, set()

    def log_weight(self, n, subset_len):
        return 0.0

    def make_strategy(self, utility, log_coefficient=None):
        return None


class DummyInfiniteSampler(IndexSampler):
    def sample_limit(self, indices):
        return None  # Indicates an infinite sampler.

    def generate(self, indices):
        while True:
            yield (0, set())

    def log_weight(self, n, subset_len):
        return 0.0

    def make_strategy(self, utility, log_coefficient=None):
        return None


def test_no_stopping_with_finite_sampler():
    r = ValuationResult.from_random(5)
    total_samples = 10
    batch_size = 3
    sampler = DummyFiniteSampler(total_samples, batch_size)
    no_stop = NoStopping(sampler=sampler)
    # Manually iterate over batches to observe sampler progress.
    gen = sampler.generate_batches(np.arange(1))
    total_seen = 0
    for _ in range(total_samples // batch_size):
        batch = list(next(gen))
        assert len(batch) == batch_size
        total_seen += len(batch)
        comp = no_stop.completion()
        np.testing.assert_allclose(comp, total_seen / sampler.total_samples)
        # Calling the criterion should always return Pending
        # and mark all indices as not converged.
        status = no_stop(r)
        assert status == Status.Pending
        np.testing.assert_equal(no_stop.converged, False)

    # Final check must trigger convergence
    _ = next(gen)
    assert no_stop(r) == Status.Converged
    np.testing.assert_equal(no_stop.converged, True)
    assert sampler.n_samples == len(sampler)
    np.testing.assert_allclose(no_stop.completion(), 1.0)


def test_no_stopping_infinite_sampler():
    sampler = DummyInfiniteSampler(batch_size=1)
    no_stop = NoStopping(sampler=sampler)

    _ = list(islice(sampler.generate_batches(np.array([0])), 10))

    # Verify that calling the criterion still returns Pending and marks no index as converged.
    result = ValuationResult.from_random(5)
    status = no_stop(result)
    assert status == Status.Pending
    assert no_stop.completion() == 0.0
    np.testing.assert_equal(no_stop.converged, False)


def test_max_samples_pending_and_convergence():
    sampler = DummyInfiniteSampler(batch_size=1)
    threshold = 10
    max_samples = MaxSamples(sampler, n_samples=threshold)
    result = ValuationResult.from_random(5)  # Create a result with 5 indices

    status = max_samples(result)
    assert status == Status.Pending
    np.testing.assert_allclose(max_samples.completion(), 0.0)
    assert not max_samples.converged.all()

    # Set sampler.n_samples below threshold.
    _ = list(islice(sampler.generate_batches(np.array([0])), 5))
    status = max_samples(result)
    assert status == Status.Pending
    np.testing.assert_allclose(max_samples.completion(), 5 / threshold)
    assert not max_samples.converged.all()

    # Set sampler.n_samples exactly equal to threshold.
    _ = list(islice(sampler.generate_batches(np.array([0])), 10))
    status = max_samples(result)
    assert status == Status.Converged
    np.testing.assert_allclose(max_samples.completion(), 1.0)
    assert max_samples.converged.all()

    # Set sampler.n_samples above threshold.
    _ = list(islice(sampler.generate_batches(np.array([0])), 15))
    status = max_samples(result)
    assert status == Status.Converged
    np.testing.assert_allclose(max_samples.completion(), 1.0)
    assert max_samples.converged.all()


def test_max_samples_str_and_invalid():
    sampler = DummyFiniteSampler(total_samples=0)
    max_samples = MaxSamples(sampler, 10)
    expected_str = f"MaxSamples({sampler.__class__.__name__}, n_samples=10)"
    assert str(max_samples) == expected_str

    with pytest.raises(ValueError):
        MaxSamples(sampler, 0)
    with pytest.raises(ValueError):
        MaxSamples(sampler, -5)

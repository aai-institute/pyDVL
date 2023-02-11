from time import sleep

import numpy as np
import pytest

from pydvl.utils import Status
from pydvl.value import ValuationResult
from pydvl.value.stopping import (
    MaxTime,
    MaxUpdates,
    MinUpdates,
    StoppingCriterion,
    make_criterion,
)


def test_stopping_criterion():
    stop = StoppingCriterion()
    assert stop.name == "StoppingCriterion"
    assert stop.modify_result is True
    with pytest.raises(NotImplementedError):
        stop(ValuationResult.empty())


def test_stopping_criterion_composition():
    c = Status.Converged
    p = Status.Pending
    f = Status.Failed

    class C(StoppingCriterion):
        def check(self, result: ValuationResult) -> Status:
            return c

    class P(StoppingCriterion):
        def check(self, result: ValuationResult) -> Status:
            return p

    class F(StoppingCriterion):
        def check(self, result: ValuationResult) -> Status:
            return f

    v = ValuationResult.empty()

    assert (~C())(v) == f
    assert (~P())(v) == c
    assert (~F())(v) == c

    assert (C() & C())(v) == c
    assert (P() | P())(v) == p

    assert (C() & P())(v) == (c & p)
    assert (C() | P())(v) == (c | p)

    assert (C() & C() & C())(v) == c
    assert (P() | P() | P())(v) == p

    assert (C() & P()).name == "Composite StoppingCriterion: C AND P"
    assert (C() | P()).name == "Composite StoppingCriterion: C OR P"
    assert (~C()).name == "Composite StoppingCriterion: NOT C"
    assert (~P()).name == "Composite StoppingCriterion: NOT P"


def test_make_criterion():
    def always_converged(result: ValuationResult) -> Status:
        return Status.Converged

    def always_pending(result: ValuationResult) -> Status:
        return Status.Pending

    def always_failed(result: ValuationResult) -> Status:
        return Status.Failed

    v = ValuationResult.empty()

    C = make_criterion(always_converged)
    P = make_criterion(always_pending)
    F = make_criterion(always_failed)

    assert C()(v) == Status.Converged
    assert P()(v) == Status.Pending
    assert F()(v) == Status.Failed

    assert C().name == "always_converged"
    assert P().name == "always_pending"
    assert F().name == "always_failed"

    assert (~C())(v) == Status.Failed
    assert (~P())(v) == Status.Converged
    assert (~F())(v) == Status.Converged

    assert (C() & C())(v) == Status.Converged
    assert (P() | P())(v) == Status.Pending
    assert (F() & F())(v) == Status.Failed
    assert (C() | F())(v) == Status.Converged


def test_minmax_updates():
    maxstop = MaxUpdates(10)
    assert maxstop.name == "MaxUpdates"
    v = ValuationResult.from_random(5)
    v.counts = np.zeros(5)
    assert maxstop(v) == Status.Pending
    v.counts += np.ones(5) * 9
    assert maxstop(v) == Status.Pending
    v.counts[0] += 1
    assert maxstop(v) == Status.Converged

    minstop = MinUpdates(10)
    assert minstop.name == "MinUpdates"
    v.counts = np.zeros(5)
    assert minstop(v) == Status.Pending
    v.counts += np.ones(5) * 9
    assert minstop(v) == Status.Pending
    v.counts[0] += 1
    assert minstop(v) == Status.Pending
    v.counts += np.ones(5)
    assert minstop(v) == Status.Converged


def test_max_time():
    v = ValuationResult.from_random(5)
    stop = MaxTime(0.3)
    assert stop(v) == Status.Pending
    sleep(0.3)
    assert stop(v) == Status.Converged


def test_history_deviation():
    pass

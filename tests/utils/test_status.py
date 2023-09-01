from pydvl.utils import Status


def test_or_status():
    """The result of bitwise or-ing two valuation statuses is given by the
    following table:

        |   | P | C | F |
        |---|---|---|---|
        | P | P | C | P |
        | C | C | C | C |
        | F | P | C | F |

    where P = Pending, C = Converged, F = Failed.
    """
    assert Status.Pending | Status.Pending == Status.Pending
    assert Status.Pending | Status.Converged == Status.Converged
    assert Status.Pending | Status.Failed == Status.Pending
    assert Status.Converged | Status.Pending == Status.Converged
    assert Status.Converged | Status.Converged == Status.Converged
    assert Status.Converged | Status.Failed == Status.Converged
    assert Status.Failed | Status.Pending == Status.Pending
    assert Status.Failed | Status.Converged == Status.Converged
    assert Status.Failed | Status.Failed == Status.Failed


def test_and_status():
    """The result of bitwise &-ing two valuation statuses is given by the
    following table:

        |   | P | C | F |
        |---|---|---|---|
        | P | P | P | F |
        | C | P | C | F |
        | F | F | F | F |

    where P = Pending, C = Converged, F = Failed.
    """
    assert Status.Pending & Status.Pending == Status.Pending
    assert Status.Pending & Status.Converged == Status.Pending
    assert Status.Pending & Status.Failed == Status.Failed
    assert Status.Converged & Status.Pending == Status.Pending
    assert Status.Converged & Status.Converged == Status.Converged
    assert Status.Converged & Status.Failed == Status.Failed
    assert Status.Failed & Status.Pending == Status.Failed
    assert Status.Failed & Status.Converged == Status.Failed
    assert Status.Failed & Status.Failed == Status.Failed


def test_not_status():
    """The result of bitwise negation of a Status is `Failed`
    if the status is `Converged`, or `Converged` otherwise:

        `~P == C, ~C == F, ~F == C`
    """
    assert ~Status.Pending == Status.Converged
    assert ~Status.Converged == Status.Failed
    assert ~Status.Failed == Status.Converged

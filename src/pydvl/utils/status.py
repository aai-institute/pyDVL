from enum import Enum


class Status(Enum):
    """Status of a computation.

    Statuses can be combined using bitwise or (``|``) and bitwise and (``&``) to
    get the status of a combined computation. For example, if we have two
    computations, one that has converged and one that has failed, then the
    combined status is ``Status.Converged | Status.Failed == Status.Converged``,
    but ``Status.Converged & Status.Failed == Status.Failed``.


    :OR:

    The result of bitwise or-ing two valuation statuses with ``|`` is given
    by the following table:

    +---+---+---+---+
    |   | P | C | F |
    +===+===+===+===+
    | P | P | C | P |
    +---+---+---+---+
    | C | C | C | C |
    +---+---+---+---+
    | F | P | C | F |
    +---+---+---+---+

    where P = Pending, C = Converged, F = Failed.

    :AND:

    The result of bitwise and-ing two valuation statuses with ``&`` is given
    by the following table:

    +---+---+---+---+
    |   | P | C | F |
    +===+===+===+===+
    | P | P | P | F |
    +---+---+---+---+
    | C | P | C | F |
    +---+---+---+---+
    | F | F | F | F |
    +---+---+---+---+

    where P = Pending, C = Converged, F = Failed.

    :NOT:

    The result of bitwise negation of a Status with ``~`` is ``Failed`` if
    the status is ``Converged``, or ``Converged`` otherwise:

        ~P == C, ~C == F, ~F == C

    :Boolean casting:

    A Status evaluates to ``True`` iff it's ``Converged`` or ``MaxIterations``

        bool(Status.Pending) == False
        bool(Status.Converged) == True
        bool(Status.Failed) == False

    :warning:
        These truth values are **inconsistent** with the usual boolean operations.
        In particular the XOR of two ``Status``es is not the same as the XOR of
        their boolean values.

    """

    Pending = "pending"
    Converged = "converged"
    MaxIterations = "maximum number of iterations reached"
    Failed = "failed"

    def __or__(self, other: "Status") -> "Status":
        if self == Status.Converged or other == Status.Converged:
            return Status.Converged
        if self == Status.Pending or other == Status.Pending:
            return Status.Pending
        if self == Status.Failed and other == Status.Failed:
            return Status.Failed
        # TODO: Should be unreachable after deleting MaxIterations:
        raise RuntimeError(f"Unexpected statuses: {self} and {other}")

    def __and__(self, other: "Status") -> "Status":
        # Careful, the order of tests matters here!
        if self == Status.Failed or other == Status.Failed:
            return Status.Failed
        if self == Status.Pending or other == Status.Pending:
            return Status.Pending
        if self == Status.Converged and other == Status.Converged:
            return Status.Converged
        # TODO: Should be unreachable after deleting MaxIterations:
        raise RuntimeError(f"Unexpected statuses: {self} and {other}")

    def __invert__(self):
        if self == Status.Converged:
            return Status.Failed
        return Status.Converged

    def __bool__(self):
        return self == Status.Converged or self == Status.MaxIterations

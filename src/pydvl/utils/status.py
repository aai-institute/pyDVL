from enum import Enum


class Status(Enum):
    Pending = "pending"
    Converged = "converged"
    MaxIterations = "maximum number of iterations reached"
    Failed = "failed"

    def __or__(self, other: "Status") -> "Status":
        """The result of bitwise or-ing two valuation statuses is given by the
        following table:

            |   | P | C | F |
            |---|---|---|---|
            | P | P | C | P |
            | C | C | C | C |
            | F | P | C | F |

        where P = Pending, C = Converged, F = Failed.
        """
        if self == Status.Converged or other == Status.Converged:
            return Status.Converged
        if self == Status.Pending or other == Status.Pending:
            return Status.Pending
        if self == Status.Failed and other == Status.Failed:
            return Status.Failed
        # TODO: Should be unreachable after deleting MaxIterations:
        raise RuntimeError(f"Unexpected statuses: {self} and {other}")

    def __and__(self, other: "Status") -> "Status":
        """The result of bitwise &-ing two valuation statuses is given by the
        following table:

            |   | P | C | F |
            |---|---|---|---|
            | P | P | P | F |
            | C | P | C | F |
            | F | F | F | F |

        where P = Pending, C = Converged, F = Failed.
        """
        if self == Status.Failed or other == Status.Failed:
            return Status.Failed
        if self == Status.Pending or other == Status.Pending:
            return Status.Pending
        if self == Status.Converged and other == Status.Converged:
            return Status.Converged
        # TODO: Should be unreachable after deleting MaxIterations:
        raise RuntimeError(f"Unexpected statuses: {self} and {other}")

    def __invert__(self):
        """The result of bitwise negation of a Status is `Failed`
        if the status is `Converged`, or `Converged` otherwise:

            `P -> C, C -> F, F -> C`
        """
        if self == Status.Converged:
            return Status.Failed
        return Status.Converged

    def __bool__(self):
        """A Status evaluates to True iff it's Converged or MaxIterations"""
        return self == Status.Converged or self == Status.MaxIterations

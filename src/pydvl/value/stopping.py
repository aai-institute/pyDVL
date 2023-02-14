import abc
from functools import update_wrapper
from time import time
from typing import Callable, Optional, Type

import numpy as np
from numpy.typing import NDArray

from pydvl.utils import Status
from pydvl.value import ValuationResult

__all__ = [
    "make_criterion",
    "StoppingCriterion",
    "StandardError",
    "MaxUpdates",
    "MaxTime",
    "HistoryDeviation",
]

StoppingCriterionCallable = Callable[[ValuationResult], Status]


class StoppingCriterion(abc.ABC):
    def __init__(self, modify_result: bool = True):
        """A composable callable object to determine whether a computation
        must stop.

        A ``StoppingCriterion`` takes a :class:`~pydvl.value.result.ValuationResult`
        and returns a :class:`~pydvl.value.result.Status~.

        :Creating stopping criteria:

        The easiest way is to declare a function implementing the interface
        :ref:`StoppingCriterionCallable` and wrap it with an instance
        of this class. Alternatively, one can inherit from this class. For some
        examples see e.g. :func:`~pydvl.value.stopping.MedianRation` and
        :func:`~pydvl.value.stopping.HistoryDeviation`

        :Composing stopping criteria:

        Objects of this type can be composed with the binary operators ``&``
        (_and_), and ``|`` (_or_), see :class:`~pydvl.utils.status.Status` for
        the truth tables. The unary operator ``~`` (_not_) is also supported.

        :param modify_result: If ``True`` the status of the input
            :class:`~pydvl.value.result.ValuationResult` is modified in place.
        """
        self.modify_result = modify_result

    @abc.abstractmethod
    def check(self, result: ValuationResult) -> Status:
        """Check whether the computation should stop."""
        ...

    @abc.abstractmethod
    def completion(self) -> float:
        """Returns a value between 0 and 1 indicating the completion of the
        computation.
        """
        ...

    @property
    def name(self):
        return type(self).__name__

    def __call__(self, result: ValuationResult) -> Status:
        if result.status is not Status.Pending:
            return result.status
        status = self.check(result)
        if self.modify_result:  # FIXME: this is not nice
            result._status = status
        return status

    def __and__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        class CompositeCriterion(StoppingCriterion):
            def check(self, result: ValuationResult) -> Status:
                return self(result) & other(result)

            @property
            def name(self):
                return f"Composite StoppingCriterion: {self.name} AND {other.name}"

            def completion(self) -> float:
                return min(self.completion(), other.completion())

        return CompositeCriterion(
            modify_result=self.modify_result or other.modify_result
        )

    def __or__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        class CompositeCriterion(StoppingCriterion):
            def check(self, result: ValuationResult) -> Status:
                return self(result) | other(result)

            @property
            def name(self):
                return f"Composite StoppingCriterion: {self.name} OR {other.name}"

            def completion(self) -> float:
                return max(self.completion(), other.completion())

        return CompositeCriterion(
            modify_result=self.modify_result or other.modify_result
        )

    def __invert__(self) -> "StoppingCriterion":
        class CompositeCriterion(StoppingCriterion):
            def check(self, result: ValuationResult) -> Status:
                return ~self(result)

            @property
            def name(self):
                return f"Composite StoppingCriterion: NOT {self.name}"

            def completion(self) -> float:
                return 1 - self.completion()

        return CompositeCriterion(modify_result=self.modify_result)


def make_criterion(fun: StoppingCriterionCallable) -> Type[StoppingCriterion]:
    """Create a new :class:`StoppingCriterion` from a function.
    Use this to enable simpler functions to be composed with bitwise operators

    :param fun: The callable to wrap.
    :return: A new subclass of :class:`StoppingCriterion`.
    """

    class WrappedCriterion(StoppingCriterion):
        def __init__(self, modify_result: bool = True):
            super().__init__(modify_result=modify_result)
            setattr(self, "check", fun)  # mypy complains if we assign to self.check
            update_wrapper(self, self.check)

        @property
        def name(self):
            return fun.__name__

        def completion(self) -> float:
            return 0.0  # FIXME: not much we can do about this...

    return WrappedCriterion


class StandardError(StoppingCriterion):
    """Compute a ratio of standard errors to values to determine convergence.

    :param threshold: A value is considered to have converged if the ratio of
        standard error to value has dropped below this value.
    """

    converged: NDArray[np.bool_]
    """A boolean array indicating whether the corresponding element has converged."""

    def __init__(self, threshold: float, modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        self.threshold = threshold
        self.converged = None  # type: ignore

    def check(self, result: ValuationResult) -> Status:
        ratios = result.stderr / result.values
        self.converged = np.where(ratios < self.threshold)
        if np.all(self.converged):
            return Status.Converged
        return Status.Pending

    def completion(self) -> float:
        return np.mean(self.converged or [0]).item()


class MaxUpdates(StoppingCriterion):
    """Terminate if any number of value updates exceeds or equals the given
    threshold.

    This checks the ``counts`` field of a
    :class:`~pydvl.value.result.ValuationResult`, i.e. the number of times that
    each index has been updated. For powerset samplers, the maximum of this
    number coincides with the maximum number of subsets sampled. For permutation
    samplers, it coincides with the number of permutations sampled.

    :param n_updates: Threshold: if ``None``, no check is performed,
        effectively creating a (never) stopping criterion that always returns
        ``Pending``.
    """

    def __init__(self, n_updates: Optional[int], modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        self.n_updates = n_updates
        self.last_max = 0

    def check(self, result: ValuationResult) -> Status:
        if self.n_updates:
            self.last_max = np.max(result.counts)
            if self.last_max >= self.n_updates:
                return Status.Converged
        return Status.Pending

    def completion(self) -> float:
        if self.n_updates is None:
            return 0.0
        return np.max(self.last_max) / self.n_updates


class MinUpdates(StoppingCriterion):
    """Terminate as soon as all value updates exceed or equal the given threshold.

    This checks the ``counts`` field of a
    :class:`~pydvl.value.result.ValuationResult`, i.e. the number of times that
    each index has been updated. For powerset samplers, the minimum of this
    number is a lower bound for the number of subsets sampled. For
    permutation samplers, it lower-bounds the amount of permutations sampled.

    :param n_updates: Threshold: if ``None``, no check is performed,
        effectively creating a (never) stopping criterion that always returns
        ``Pending``.
    """

    def __init__(self, n_updates: Optional[int], modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        self.n_updates = n_updates
        self.last_min = 0

    def check(self, result: ValuationResult) -> Status:
        if self.n_updates is not None:
            self.last_min = np.min(result.counts)
            if self.last_min >= self.n_updates:
                return Status.Converged
        return Status.Pending

    def completion(self) -> float:
        if self.n_updates is None:
            return 0.0
        return np.min(self.last_min) / self.n_updates


class MaxTime(StoppingCriterion):
    """Terminate if the computation time exceeds the given number of seconds.

    Checks the elapsed time since construction

    :param seconds: Threshold: The computation is terminated if the elapsed time
        between object construction and a check exceeds this value. If ``None``,
        no check is performed, effectively creating a (never) stopping criterion
        that always returns ``Pending``.
    """

    def __init__(self, seconds: Optional[float], modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        self.max_seconds = seconds
        if self.max_seconds <= 0:
            raise ValueError("Number of seconds for MaxTime must be positive")
        self.start = time()

    def check(self, result: ValuationResult) -> Status:
        if self.max_seconds is not None and time() > self.start + self.max_seconds:
            return Status.Converged
        return Status.Pending

    def completion(self) -> float:
        if self.max_seconds is None:
            return 0.0
        return (time() - self.start) / self.max_seconds


class HistoryDeviation(StoppingCriterion):
    r"""A simple check for relative distance to a previous step in the
    computation.

    The method used by :footcite:t:`ghorbani_data_2019` computes the relative
    distances between the current values $v_i^t$ and the values at the previous
    checkpoint $v_i^{t-\tau}$. If the sum is below a given threshold, the
    computation is terminated.

    $$\sum_{i=1}^n \frac{\left| v_i^t - v_i^{t-\tau} \right|}{v_i^t} <
    \epsilon.$$

    When the denominator is zero, the summand is set to the value of $v_i^{
    t-\tau}$.

    This implementation is slightly generalised to allow for different number of
    updates to individual indices, as happens with powerset samplers instead of
    permutations. Every subset of indices that is found to converge can be
    pinned
    to that state. Once all indices have converged the method has converged.

    .. warning::
       This criterion is meant for the reproduction of the results in the paper,
       but we do not recommend using it in practice.

    :param n_steps: Checkpoint values every so many updates and use these saved
        values to compare.
    :param rtol: Relative tolerance for convergence ($\epsilon$ in the formula).
    :param pin_converged: If ``True``, once an index has converged, it is pinned
    """

    memory: NDArray[np.float_]
    converged: NDArray[np.bool_]

    def __init__(
        self,
        n_steps: int,
        rtol: float,
        pin_converged: bool = True,
        modify_result: bool = True,
    ):
        super().__init__(modify_result=modify_result)
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if rtol <= 0 or rtol >= 1:
            raise ValueError("rtol must be in (0, 1)")

        self.n_steps = n_steps
        self.rtol = rtol
        self.update_op = np.logical_or if pin_converged else np.logical_and
        self.memory = None  # type: ignore
        self.converged = None  # type: ignore

    def check(self, r: ValuationResult) -> Status:
        if self.memory is None:
            self.memory = np.full((len(r.values), self.n_steps + 1), np.inf)
            self.converged = np.full(len(r), False)
            return Status.Pending

        # shift left: last column is the last set of values
        self.memory = np.concatenate(
            [self.memory[:, 1:], r.values.reshape(-1, 1)], axis=1
        )

        # Look at indices that have been updated more than n_steps times
        ii = np.where(r.counts > self.n_steps)
        if len(ii) > 0:
            curr = self.memory[:, -1]
            saved = self.memory[:, 0]
            diffs = np.abs(curr[ii] - saved[ii])
            quots = np.divide(diffs, curr[ii], out=diffs, where=curr[ii] != 0)
            # quots holds the quotients when the denominator is non-zero, and
            # the absolute difference, which is just the memory, otherwise.
            if np.mean(quots) < self.rtol:
                self.converged = self.update_op(self.converged, ii)
                if np.all(self.converged):
                    return Status.Converged
        return Status.Pending

    def completion(self) -> float:
        return np.mean(self.converged or [0]).item()

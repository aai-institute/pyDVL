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
    "MedianRatio",
    "MaxUpdates",
    "MaxTime",
    "HistoryDeviation",
]

StoppingCriterionCallable = Callable[[ValuationResult], Status]


class StoppingCriterion:
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
        (_and_), ``^`` (_xor_) and ``|`` (_or_),
        see :class:`~pydvl.utils.status.Status` for the truth tables. The unary
        operator ``~`` (_not_) is also supported.

        :param fun: A callable to wrap into a composable object. Use this to
            enable simpler functions to be composed with the operators described
            above.
        :param modify_result: If ``True`` the status of the input
            :class:`~pydvl.value.result.ValuationResult` is modified in place.
        """
        self.modify_result = modify_result

    def check(self, result: ValuationResult) -> Status:
        raise NotImplementedError

    @property
    def name(self):
        return type(self).__name__

    def __call__(self, result: ValuationResult) -> Status:
        status = self.check(result)
        if self.modify_result:  # FIXME: this is not nice
            result._status = status
        return status

    def __and__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        def fun(result: ValuationResult):
            return self(result) & other(result)

        fun.__name__ = f"Composite StoppingCriterion: {self.name} AND {other.name}"
        return make_criterion(fun)(
            modify_result=self.modify_result or other.modify_result
        )

    def __or__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        def fun(result: ValuationResult):
            return self(result) | other(result)

        fun.__name__ = f"Composite StoppingCriterion: {self.name} OR {other.name}"
        return make_criterion(fun)(
            modify_result=self.modify_result or other.modify_result
        )

    def __invert__(self) -> "StoppingCriterion":
        def fun(result: ValuationResult):
            return ~self(result)

        fun.__name__ = f"Composite StoppingCriterion: NOT {self.name}"
        return make_criterion(fun)(modify_result=self.modify_result)


def make_criterion(fun: StoppingCriterionCallable) -> Type[StoppingCriterion]:
    """Create a new :class:`StoppingCriterion` from a function."""

    class WrappedCriterion(StoppingCriterion):
        def __init__(self, modify_result: bool = True):
            super().__init__(modify_result=modify_result)
            setattr(self, "check", fun)  # mypy complains if we assign to self.check
            update_wrapper(self, self.check)

        @property
        def name(self):
            return fun.__name__

    return WrappedCriterion


class MedianRatio(StoppingCriterion):
    """Compute a ratio of median errors to values to determine convergence.

    :param threshold: Converged if the ratio of median standard error to median
        of value has dropped below this value.
    """

    def __init__(self, threshold: float, modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        self.threshold = threshold

    def check(self, result: ValuationResult) -> Status:
        ratio = np.median(result.stderr) / np.median(result.values)
        if ratio < self.threshold:
            return Status.Converged
        return Status.Pending


class MaxUpdates(StoppingCriterion):
    """Terminate if any number of value updates exceeds or equals the given
    threshold.

    This checks the ``counts`` field of a :class:`~pydvl.value.result.ValuationResult`, i.e. the number of times that
    each index has been updated. For powerset samplers, the maximum of this
    number coincides with the maximum number of subsets sampled. For
    permutation samplers, it coincides with the number of permutations sampled.

    :param n_updates: Threshold: if ``None``, no check is performed,
        effectively creating a (never) stopping criterion that always returns
        ``Pending``.
    """

    def __init__(self, n_updates: Optional[int], modify_result: bool = True):
        self.n_updates = n_updates
        super().__init__(modify_result=modify_result)

    def check(self, result: ValuationResult) -> Status:
        if self.n_updates is not None:
            try:
                if np.max(result.counts) >= self.n_updates:
                    return Status.Converged
            except AttributeError:
                raise ValueError("ValuationResult didn't contain a `counts` attribute")

        return Status.Pending


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
        self.n_updates = n_updates
        super().__init__(modify_result=modify_result)

    def check(self, result: ValuationResult) -> Status:
        if self.n_updates is not None:
            try:
                if np.min(result.counts) >= self.n_updates:
                    return Status.Converged
            except AttributeError:
                raise ValueError("ValuationResult didn't contain a `counts` attribute")

        return Status.Pending


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
        self.start = time()

    def check(self, result: ValuationResult) -> Status:
        if self.max_seconds is not None and time() > self.start + self.max_seconds:
            return Status.Converged
        return Status.Pending


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
    checked: NDArray[np.bool_]

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
        self.pin_converged = pin_converged
        self.memory = None  # type: ignore
        self.checked = None  # type: ignore

    def check(self, r: ValuationResult) -> Status:
        if self.memory is None:
            self.memory = np.full((len(r.values), self.n_steps + 1), np.inf)
            self.checked = np.full(len(r), False)
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
                if not self.pin_converged:
                    self.checked = np.full(len(r), False)
                self.checked[ii] = True
                if np.all(self.checked):
                    return Status.Converged
        return Status.Pending

    # def tmp(self):
    #     # shift left: last column is the last set of values
    #     memory = np.concatenate([memory[:, 1:], values.reshape(-1, 1)], axis=1)
    #     if np.all(counts > n_steps):
    #         diff = memory @ coefficients
    #         passing_ratio = np.count_nonzero(diff < atol) / len(diff)
    #         if passing_ratio >= values_ratio:
    #             return ValuationStatus.Converged
    #     return ValuationStatus.Pending

from functools import update_wrapper
from time import time
from typing import Callable, Optional, Type

import numpy as np

from pydvl.utils import Status
from pydvl.value import ValuationResult

__all__ = [
    "make_criterion",
    "StoppingCriterion",
    "MedianRatio",
    "MaxIterations",
    "MaxTime",
    "HistoryDeviation",
]

StoppingCriterionCallable = Callable[[ValuationResult], Status]


class StoppingCriterion:
    def __init__(self, modify_results: bool = True):
        """A composable callable object to determine whether a computation
        must stop.

        ``StoppingCriterion``s take a
        :class:`~pydvl.value.results.ValuationResult`
        and return a :class:`~pydvl.value.results.Status~.

        :Creating StoppingCriterions:

        The easiest way is to declare a function implementing the interface
        :ref:`StoppingCriterionCallable` and wrap it with an instance
        of this class. Alternatively, one can inherit from this class. For some
        examples see e.g. :func:`~pydvl.value.stopping.MedianRation` and
        :func:`~pydvl.value.stopping.HistoryDeviation`

        :Composing stopping criteria:

        Objects of this type can be composed with the binary operators ``&``
        (_and_), ``^`` (_xor_) and ``|`` (_or_),
        see :class:`~pydvl.value.results.Status` for the truth tables. The unary
        operator ``~`` (_not_) is also supported.

        :param fun: A callable to wrap into a composable object. Use this to
            enable simpler functions to be composed with the operators described
            above.
        :param modify_results: If ``True`` the status of the input
            :class:`~pydvl.value.results.ValuationResult` is modified in place.
        """
        self.modify_results = modify_results

    def check(self, results: ValuationResult) -> Status:
        raise NotImplementedError

    @property
    def name(self):
        return type(self).__name__

    def __call__(self, results: ValuationResult) -> Status:
        status = self.check(results)
        if self.modify_results:  # FIXME: this is not nice
            results._status = status
        return status

    def __and__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        def fun(results: ValuationResult):
            return self(results) & other(results)

        fun.__name__ = f"Composite StoppingCriterion: {self.name} AND {other.name}"
        return make_criterion(fun)(
            modify_results=self.modify_results or other.modify_results
        )

    def __or__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        def fun(results: ValuationResult):
            return self(results) | other(results)

        fun.__name__ = f"Composite StoppingCriterion: {self.name} OR {other.name}"
        return make_criterion(fun)(
            modify_results=self.modify_results or other.modify_results
        )

    def __invert__(self) -> "StoppingCriterion":
        def fun(results: ValuationResult):
            return ~self(results)

        fun.__name__ = f"Composite StoppingCriterion: NOT {self.name}"
        return make_criterion(fun)(modify_results=self.modify_results)

    def __xor__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        def fun(results: ValuationResult):
            a = self(results)
            b = other(results)
            return (a & ~b) | (~a & b)

        fun.__name__ = f"Composite StoppingCriterion: {self.name} XOR {other.name}"
        return make_criterion(fun)(
            modify_results=self.modify_results or other.modify_results
        )


def make_criterion(fun: StoppingCriterionCallable) -> Type[StoppingCriterion]:
    """Create a new :class:`StoppingCriterion` from a function."""

    class WrappedCriterion(StoppingCriterion):
        def __init__(self, modify_results: bool = True):
            super().__init__(modify_results=modify_results)
            self.check = fun
            update_wrapper(self, self.check)

        @property
        def name(self):
            return fun.__name__

    return WrappedCriterion


class MedianRatio(StoppingCriterion):
    """Compute a ratio of median errors to values to determine convergence.

    :param threshold: Converged if the ratio of median standard
        error to median of value has dropped below this value.
    """

    def __init__(self, threshold: float, modify_results: bool = True):
        super().__init__(modify_results=modify_results)
        self.threshold = threshold

    def check(self, results: ValuationResult) -> Status:
        ratio = np.median(results.stderr) / np.median(results.values)
        if ratio < self.threshold:
            return Status.Converged
        return Status.Pending


class MaxIterations(StoppingCriterion):
    """Terminate if the number of value updates exceeds the given number.

    This checks the ``counts`` field of a
    :class:`~pydvl.value.results.ValuationResult`, i.e. the number of times that
    each index has been updated. For powerset samplers, the maximum of this
    number coincides with the maximum number of subsets sampled. For
    permutation samplers, it coincides with the number of permutations sampled.

    :param n_iterations: Threshold: if ``None``, no check is performed,
        effectively creating a (never) stopping criterion that always returns
        ``Pending``.
    """

    def __init__(self, n_iterations: Optional[int], modify_results: bool = True):
        self.n_iterations = n_iterations
        super().__init__(modify_results=modify_results)

    def check(self, results: ValuationResult) -> Status:
        if self.n_iterations is not None:
            try:
                if np.max(results.counts) > self.n_iterations:
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

    def __init__(self, seconds: Optional[float], modify_results: bool = True):
        super().__init__(modify_results=modify_results)
        self.max_seconds = seconds
        self.start = time()

    def check(self, results: ValuationResult) -> Status:
        if self.max_seconds is not None and time() > self.start + self.max_seconds:
            return Status.Converged
        return Status.Pending


class HistoryDeviation(StoppingCriterion):
    """Ghorbani et al.'s check for relative distance to a previous step in the
     computation.

    This is slightly generalised to allow for different number of updates to
    individual indices, as happens with powerset samplers instead of
    permutations. Every subset of indices that is found to converge is pinned to
    that state. Once all indices have converged the method has converged.

    :param n_samples: Number of values in the result objects.
    :param n_steps: Checkpoint values every so many updates and use these saved
        values to compare.
    :param rtol: Relative tolerance for convergence.
    """

    def __init__(
        self, n_samples: int, n_steps: int, rtol: float, modify_results: bool = True
    ):
        super().__init__(modify_results=modify_results)
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.rtol = rtol
        self.memory = np.zeros(n_samples)
        self.converged = np.array([False] * n_samples)

    def check(self, r: ValuationResult) -> Status:
        if r.counts.max() == 0:  # safeguard against reuse of the criterion
            self.memory = np.zeros(self.n_samples)
            self.converged = np.array([False] * self.n_samples)
            return Status.Pending
        # Look at indices that have been updated n_steps times since last save
        # For permutation samplers, this should be all indices, every n_steps
        ii = np.where(r.counts % self.n_steps == 0)
        if len(ii) > 0:
            if (
                np.abs((r.values[ii] - self.memory[ii]) / r.values[ii]).mean()
                < self.rtol
            ):
                self.converged[ii] = True
                if np.all(self.converged):
                    return Status.Converged
            self.memory[ii] = r.values[ii]
        return Status.Pending

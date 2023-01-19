from functools import update_wrapper
from typing import Callable, cast

import numpy as np

from pydvl.utils import Status
from pydvl.value import ValuationResult

__all__ = ["ConvergenceCheck", "median_ratio", "max_iterations", "history_deviation"]

ConvergenceCheckCallable = Callable[[ValuationResult], Status]


class ConvergenceCheck:
    _fun: ConvergenceCheckCallable

    def __init__(self, fun: ConvergenceCheckCallable):
        """A composable callable object to determine whether a computation must stop.

        ``ConvergenceCheck``s take a :class:`~pydvl.value.results.ValuationResult`
        and return a :class:`~pydvl.value.results.Status~.

        :Creating ConvergenceChecks:

        The easiest way is to declare a function implementing the
        interface :ref:`ConvergenceCheckCallable` and wrap it with this class

        For more realistic examples see
        e.g. :func:`~pydvl.value.convergence.median_ratio`
        or :func:`~pydvl.value.convergence.history_deviation`

        :Composing ConvergenceChecks:

        Objects of this type can be composed with the binary operators ``&``
        (_and_), ``^`` (_xor_) and ``|`` (_or_),
        see :class:`~pydvl.value.results.Status` for the truth tables. The unary
        operator ``~`` (_not_) is also supported.

        :param fun: A callable to wrap into a composable object. Use this to
            enable simpler functions to be composed with the operators described
            above.

        """
        self._fun = fun
        update_wrapper(self, self._fun)

    @property
    def name(self):
        return self._fun.__name__

    def __call__(self, results: ValuationResult) -> Status:
        return self._fun(results)

    def __and__(self, other: "ConvergenceCheck") -> "ConvergenceCheck":
        def fun(results: ValuationResult):
            return self(results) & other(results)

        fun.__name__ = f"Composite ConvergenceCheck: {self.name} AND {other.name}"
        return ConvergenceCheck(cast(ConvergenceCheckCallable, fun))

    def __or__(self, other: "ConvergenceCheck") -> "ConvergenceCheck":
        def fun(results: ValuationResult):
            return self(results) | other(results)

        fun.__name__ = f"Composite ConvergenceCheck: {self.name} OR {other.name}"
        return ConvergenceCheck(cast(ConvergenceCheckCallable, fun))

    def __invert__(self) -> "ConvergenceCheck":
        def fun(results: ValuationResult):
            return ~self(results)

        fun.__name__ = f"Composite ConvergenceCheck: NOT {self.name}"
        return ConvergenceCheck(cast(ConvergenceCheckCallable, fun))

    def __xor__(self, other: "ConvergenceCheck") -> "ConvergenceCheck":
        def fun(results: ValuationResult):
            a = self(results)
            b = other(results)
            return (a & ~b) | (~a & b)

        fun.__name__ = f"Composite ConvergenceCheck: {self.name} XOR {other.name}"
        return ConvergenceCheck(cast(ConvergenceCheckCallable, fun))


def median_ratio(threshold: float) -> ConvergenceCheck:
    """Ratio of medians for convergence.

    :param threshold: Converged if the ratio of median standard
        error to median of value has dropped below this value.
    :return: The convergence check
    """

    def median_ratio_check(results: ValuationResult) -> Status:
        ratio = np.median(results.stderr) / np.median(results.values)
        if ratio < threshold:
            return Status.Converged
        return Status.Pending

    return ConvergenceCheck(median_ratio_check)


def max_iterations(n_iterations: int) -> ConvergenceCheck:
    """Terminate if the number of iterations exceeds the given number.

    This checks the ``counts`` field of a
    :class:`~pydvl.value.results.ValuationResult`. For powerset samplers, the
    maximum number of times that a particular index has been updated coincides
    with the maximum number of subsets sampled. For permutation samplers, it
    coincides with the number of permutations sampled.

    :param n_iterations: threshold.
    :return: The convergence check
    """
    _max_iterations = n_iterations

    def max_iterations_check(results: ValuationResult) -> Status:
        try:
            if np.max(results.counts) > _max_iterations:
                return Status.Converged
        except AttributeError:
            raise ValueError("ValuationResult didn't contain a `counts` attribute")

        return Status.Pending

    return ConvergenceCheck(max_iterations_check)


def history_deviation(n_samples: int, n_steps: int, atol: float) -> ConvergenceCheck:
    """Ghorbani et al.'s check for relative distance to a previous step in the
     computation.

    This is slightly generalised to allow for different number of updates to
    individual indices, as happens with powerset samplers instead of
    permutations. Every subset of indices that is found to converge is pinned to
    that state. Once all indices have converged the method has converged.

    :param n_samples: Number of values in the result objects.
    :param n_steps: Checkpoint values every so many updates and use these saved
        values to compare.
    :param atol: Absolute tolerance for convergence.
    :return: The convergence check
    """
    memory = np.zeros(n_samples)
    converged = np.array([False] * n_samples)

    def history_deviation_check(r: ValuationResult) -> Status:
        nonlocal memory
        if r.counts.max() == 0:  # safeguard against reuse of the criterion
            memory = np.zeros(n_samples)
            return Status.Pending
        # Look at indices that have been updated n_steps times since last save
        # For permutation samplers, this should be all indices, every n_steps
        ii = np.where(r.counts % n_steps == 0)
        if len(ii) > 0:
            if np.abs((r.values[ii] - memory[ii]) / r.values[ii]).mean() < atol:
                converged[ii] = True
                if np.all(converged):
                    return Status.Converged
            memory[ii] = r.values[ii]
        return Status.Pending

    return ConvergenceCheck(history_deviation_check)

from functools import update_wrapper
from typing import Callable, Optional, cast

import numpy as np

from pydvl.utils import Status
from pydvl.value import ValuationResult

__all__ = ["StoppingCriterion", "median_ratio", "max_iterations", "history_deviation"]

StoppingCriterionCallable = Callable[[ValuationResult], Status]


class StoppingCriterion:
    _fun: StoppingCriterionCallable

    def __init__(self, fun: StoppingCriterionCallable, inplace: bool = True):
        """A composable callable object to determine whether a computation must stop.

        ``StoppingCriterion``s take a :class:`~pydvl.value.results.ValuationResult`
        and return a :class:`~pydvl.value.results.Status~.

        :Creating StoppingCriterions:

        The easiest way is to declare a function implementing the
        interface :ref:`StoppingCriterionCallable` and wrap it with this class

        For more realistic examples see
        e.g. :func:`~pydvl.value.stopping.median_ratio`
        or :func:`~pydvl.value.stopping.history_deviation`

        :Composing stopping criteria:

        Objects of this type can be composed with the binary operators ``&``
        (_and_), ``^`` (_xor_) and ``|`` (_or_),
        see :class:`~pydvl.value.results.Status` for the truth tables. The unary
        operator ``~`` (_not_) is also supported.

        :param fun: A callable to wrap into a composable object. Use this to
            enable simpler functions to be composed with the operators described
            above.
        :param inplace: If ``True`` the status of the input ValuationResult is
            modified in place.
        """
        self.inplace = inplace
        self._fun = fun
        update_wrapper(self, self._fun)

    @property
    def name(self):
        return self._fun.__name__

    def __call__(self, results: ValuationResult) -> Status:
        status = self._fun(results)
        if self.inplace:  # FIXME: this is not nice
            results._status = status
        return status

    def __and__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        def fun(results: ValuationResult):
            return self(results) & other(results)

        fun.__name__ = f"Composite StoppingCriterion: {self.name} AND {other.name}"
        return StoppingCriterion(cast(StoppingCriterionCallable, fun))

    def __or__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        def fun(results: ValuationResult):
            return self(results) | other(results)

        fun.__name__ = f"Composite StoppingCriterion: {self.name} OR {other.name}"
        return StoppingCriterion(cast(StoppingCriterionCallable, fun))

    def __invert__(self) -> "StoppingCriterion":
        def fun(results: ValuationResult):
            return ~self(results)

        fun.__name__ = f"Composite StoppingCriterion: NOT {self.name}"
        return StoppingCriterion(cast(StoppingCriterionCallable, fun))

    def __xor__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        def fun(results: ValuationResult):
            a = self(results)
            b = other(results)
            return (a & ~b) | (~a & b)

        fun.__name__ = f"Composite StoppingCriterion: {self.name} XOR {other.name}"
        return StoppingCriterion(cast(StoppingCriterionCallable, fun))


def median_ratio(threshold: float) -> StoppingCriterion:
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

    return StoppingCriterion(median_ratio_check)


def max_iterations(n_iterations: Optional[int]) -> StoppingCriterion:
    """Terminate if the number of iterations exceeds the given number.

    This checks the ``counts`` field of a
    :class:`~pydvl.value.results.ValuationResult`. For powerset samplers, the
    maximum number of times that a particular index has been updated coincides
    with the maximum number of subsets sampled. For permutation samplers, it
    coincides with the number of permutations sampled.

    :param n_iterations: threshold. If ``None``, no check is performed,
        effectively creating a convergence check that always returns ``Pending``.
    :return: The convergence check
    """
    if n_iterations is None:
        return StoppingCriterion(lambda _: Status.Pending)

    _max_iterations = n_iterations

    def max_iterations_check(results: ValuationResult) -> Status:
        try:
            if np.max(results.counts) > _max_iterations:
                return Status.Converged
        except AttributeError:
            raise ValueError("ValuationResult didn't contain a `counts` attribute")

        return Status.Pending

    return StoppingCriterion(max_iterations_check)


def history_deviation(n_samples: int, n_steps: int, rtol: float) -> StoppingCriterion:
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
    :return: The convergence check
    """
    # Yuk... using static vars just to avoid nonlocal
    # An abstract class and maybe a factory would probably be better
    def history_deviation_check(r: ValuationResult) -> Status:
        if r.counts.max() == 0:  # safeguard against reuse of the criterion
            history_deviation_check.memory = np.zeros(n_samples)
            history_deviation_check.converged = np.array([False] * n_samples)
            return Status.Pending
        # Look at indices that have been updated n_steps times since last save
        # For permutation samplers, this should be all indices, every n_steps
        ii = np.where(r.counts % n_steps == 0)
        if len(ii) > 0:
            if (
                np.abs(
                    (r.values[ii] - history_deviation_check.memory[ii]) / r.values[ii]
                ).mean()
                < rtol
            ):
                history_deviation_check.converged[ii] = True
                if np.all(history_deviation_check.converged):
                    return Status.Converged
            history_deviation_check.memory[ii] = r.values[ii]
        return Status.Pending

    history_deviation_check.memory = np.zeros(n_samples)
    history_deviation_check.converged = np.array([False] * n_samples)
    return StoppingCriterion(history_deviation_check)

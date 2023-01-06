""" Stopping criteria for semi-values

"""

from functools import update_wrapper
from typing import Callable, cast

import numpy as np
from numpy._typing import NDArray

from pydvl.value import ValuationStatus

StoppingCriterionCallable = Callable[
    [int, NDArray[np.float_], NDArray[np.float_], NDArray[np.int_]], ValuationStatus,
]


class StoppingCriterion:
    _fun: StoppingCriterionCallable

    def __init__(self, fun: StoppingCriterionCallable):
        """A composable callable to determine whether a semi-value computation
        must stop.

        Stopping criteria can be composed with the binary
        operators ``&`` (_and_), ``^`` (_xor_) and ``|`` (_or_),
        see :class:`~pydvl.value.results.ValuationStatus` for the truth tables.

        :param fun: A callable to wrap into a composable object.

        """
        self._fun = fun
        update_wrapper(self, fun)

    def __call__(
        self,
        step: int,
        values: NDArray[np.float_],
        variances: NDArray[np.float_],
        counts: NDArray[np.int_],
    ) -> ValuationStatus:
        return self._fun(step, values, variances, counts)

    def __and__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        # Using variadic args to avoid repeating all args
        def fun(*args, **kwargs):
            return self(*args, **kwargs) & other(*args, **kwargs)

        fun.__name__ = (
            f"Composite StoppingCriterion: {self.__name__} AND {other.__name__}"
        )
        return StoppingCriterion(cast(StoppingCriterionCallable, fun))

    def __or__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        def fun(*args, **kwargs):
            return self(*args, **kwargs) | other(*args, **kwargs)

        fun.__name__ = (
            f"Composite StoppingCriterion: {self.__name__} OR {other.__name__}"
        )
        return StoppingCriterion(cast(StoppingCriterionCallable, fun))

    def __invert__(self) -> "StoppingCriterion":
        def fun(*args, **kwargs):
            return ~self(*args, **kwargs)

        fun.__name__ = f"Composite StoppingCriterion: NOT {self.__name__}"
        return StoppingCriterion(cast(StoppingCriterionCallable, fun))

    def __xor__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        def fun(*args, **kwargs):
            a = self(*args, **kwargs)
            b = other(*args, **kwargs)
            return (a & ~b) | (~a & b)

        fun.__name__ = (
            f"Composite StoppingCriterion: {self.__name__} XOR {other.__name__}"
        )
        return StoppingCriterion(cast(StoppingCriterionCallable, fun))


def max_samples_criterion(max_samples: np.int_) -> StoppingCriterion:
    def check_max_samples(step: int, *args, **kwargs) -> ValuationStatus:
        if step >= max_samples:
            return ValuationStatus.Converged
        return ValuationStatus.Pending

    return StoppingCriterion(cast(StoppingCriterionCallable, check_max_samples))


def min_updates_criterion(min_updates: int, values_ratio: float) -> StoppingCriterion:
    """Checks whether a given fraction of all values has been updated at
    least ``min_updates`` times.

    :param min_updates: Maximal amount of updates for each value
    :param values_ratio: Amount of values that must fulfill the criterion
    :return: :attr:`~pydvl.value.results.ValuationStatus.Converged` if at least
    a fraction of ``values_ratio`` of the values has been updated
    ``min_updates`` times, :attr:`~pydvl.value.results.ValuationStatus.Pending`
    otherwise.

    """

    def check_min_updates(*args, counts: NDArray[np.int_], **kwargs) -> ValuationStatus:
        if np.count_nonzero(counts >= min_updates) / len(counts) >= values_ratio:
            return ValuationStatus.Converged
        return ValuationStatus.Pending

    return StoppingCriterion(cast(StoppingCriterionCallable, check_min_updates))


def stderr_criterion(eps: float, values_ratio: float) -> StoppingCriterion:
    r"""Checks that the standard error of the values is below a threshold.

    A value $v_i$ is considered to be below the threshold if the associated
    standard error $s_i = \sqrt{\var{v_ii}/n}$ fulfills:

    $$ s_i \lt | \eps v_i | . $$

    In other words, the computation of the value for sample $x_i$ is considered
    complete once the estimator for the standard error is within a fraction
    $\eps$ of the value.

    .. fixme::
       This ad-hoc will fail if the distribution of utilities for an index has
       high variance. We need something better, taking 1st or maybe 2nd order
       info into account.

    :param eps: Threshold multiplier for the values
    :param values_ratio: Amount of values that must fulfill the criterion
    :return: A convergence criterion
        returning :attr:`~pydvl.value.results.ValuationStatus.Converged` if at
        least a fraction of `values_ratio` of the values has standard error
        below the threshold.
    """

    def check_stderr(
        step: int,
        values: NDArray[np.float_],
        variances: NDArray[np.float_],
        counts: NDArray[np.int_],
    ) -> ValuationStatus:
        if len(values) == 0:
            raise ValueError("Empty values array")
        if len(values) != len(variances) or len(values) != len(counts):
            raise ValueError("Mismatching array lengths")

        if np.any(counts == 0):
            return ValuationStatus.Pending

        passing_ratio = np.count_nonzero(
            np.sqrt(variances / counts) <= np.abs(eps * values)
        ) / len(values)
        if passing_ratio >= values_ratio:
            return ValuationStatus.Converged
        return ValuationStatus.Pending

    return StoppingCriterion(check_stderr)


def finite_difference_criterion(
    n_steps: int, n_values: int, atol: float, values_ratio: float
) -> StoppingCriterion:
    """Uses a discrete 1st derivative to define convergence.

    A value computation is considered to have converged if the backward finite
    difference computed using the last ``n_steps`` values is ``atol`` close to
    zero.

    .. todo::
       Allow arbitrary choices of ``n_steps``.

    :param n_steps: number of values to use for finite difference
    :param n_values: number of items in the values array (number of samples)
    :param atol: absolute tolerance for the finite difference.
    :param values_ratio: amount of values that must fulfill the criterion
    """
    if n_steps != 7:
        raise NotImplementedError("Generalised finite difference not implemented")
    # FIXME: these are coefficients for the grid with x = 0,1,2,3,4,5,6
    coefficients = np.array([49 / 20, -6, 15 / 2, -20 / 3, 15 / 4, -6 / 5, 1 / 6])

    # FIXME: maybe I need objects after all... :/
    memory = np.zeros(shape=(n_values, n_steps))

    def check_finite_differences(
        step: int,
        values: NDArray[np.float_],
        variances: NDArray[np.float_],
        counts: NDArray[np.int_],
    ) -> ValuationStatus:
        nonlocal memory
        if step == 0:  # safeguard against reuse of the criterion
            memory = np.zeros(shape=(n_values, n_steps))
        # shift left: last column is the last set of values
        memory = np.concatenate([memory[:, 1:], values.reshape(-1, 1)], axis=1)
        if np.all(counts > n_steps):
            diff = memory @ coefficients
            passing_ratio = np.count_nonzero(diff < atol) / len(diff)
            if passing_ratio >= values_ratio:
                return ValuationStatus.Converged
        return ValuationStatus.Pending

    return StoppingCriterion(check_finite_differences)


def ghorbani_criterion(
    n: int, n_steps: int = 100, atol: float = 0.05
) -> StoppingCriterion:
    """Implements the stopping criterion for TMC and G-Shapley used in
    :footcite:t:`ghorbani_data_2019`.

    .. rubric:: References

    .. footbibliography::

    """
    memory = np.zeros(shape=(n,))

    def check_ghorbani_criterion(
        step: int,
        values: NDArray[np.float_],
        variances: NDArray[np.float_],
        counts: NDArray[np.int_],
    ) -> ValuationStatus:
        nonlocal memory
        if step == 0:  # safeguard against reuse of the criterion
            memory = np.zeros(shape=(n,))
            return ValuationStatus.Pending
        if step % 100 == 0:
            if np.abs((values - memory) / values).mean() < atol:
                return ValuationStatus.Converged
            memory[:] = values[:]
        return ValuationStatus.Pending

    return StoppingCriterion(check_ghorbani_criterion)

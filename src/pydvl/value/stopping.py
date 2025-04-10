r"""
Stopping criteria for value computations.

This module provides a basic set of stopping criteria, like
[MaxUpdates][pydvl.value.stopping.MaxUpdates],
[MaxTime][pydvl.value.stopping.MaxTime], or
[HistoryDeviation][pydvl.value.stopping.HistoryDeviation] among others. These
can behave in different ways depending on the context. For example,
[MaxUpdates][pydvl.value.stopping.MaxUpdates] limits
the number of updates to values, which depending on the algorithm may mean a
different number of utility evaluations or imply other computations like solving
a linear or quadratic program.

Stopping criteria are callables that are evaluated on a
[ValuationResult][pydvl.value.result.ValuationResult] and return a
[Status][pydvl.utils.status.Status] object. They can be combined using boolean
operators.

## How convergence is determined

Most stopping criteria keep track of the convergence of each index separately
but make global decisions based on the overall convergence of some fraction of
all indices. For example, if we have a stopping criterion that checks whether
the standard error of 90% of values is below a threshold, then methods will keep
updating **all** indices until 90% of them have converged, irrespective of the
quality of the individual estimates, and *without freezing updates* for indices
along the way as values individually attain low standard error.

This has some practical implications, because some values do tend to converge
sooner than others. For example, assume we use the criterion
`AbsoluteStandardError(0.02) | MaxUpdates(1000)`. Then values close to 0 might
be marked as "converged" rather quickly because they fulfill the first
criterion, say after 20 iterations, despite being poor estimates. Because other
indices take much longer to have low standard error and the criterion is a
global check, the "converged" ones keep being updated and end up being good
estimates. In this case, this has been beneficial, but one might not wish for
converged values to be updated, if one is sure that the criterion is adequate
for individual values.

[Semi-value methods][pydvl.value.semivalues] include a parameter
`skip_converged` that allows to skip the computation of values that have
converged. The way to avoid doing this too early is to use a more stringent
check, e.g. `AbsoluteStandardError(1e-3) | MaxUpdates(1000)`. With
`skip_converged=True` this check can still take less time than the first one,
despite requiring more iterations for some indices.


## Choosing a stopping criterion

The choice of a stopping criterion greatly depends on the algorithm and the
context. A safe bet is to combine a [MaxUpdates][pydvl.value.stopping.MaxUpdates]
or a [MaxTime][pydvl.value.stopping.MaxTime] with a
[HistoryDeviation][pydvl.value.stopping.HistoryDeviation] or an
[AbsoluteStandardError][pydvl.value.stopping.AbsoluteStandardError]. The former
will ensure that the computation does not run for too long, while the latter
will try to achieve results that are stable enough. Note however that if the
threshold is too strict, one will always end up running until a maximum number
of iterations or time. Also keep in mind that different values converge at
different times, so you might want to use tight thresholds and `skip_converged`
as described above for semi-values.


??? Example
    ```python
    from pydvl.value import AbsoluteStandardError, MaxUpdates, compute_banzhaf_semivalues

    utility = ...  # some utility object
    criterion = AbsoluteStandardError(threshold=1e-3, burn_in=32) | MaxUpdates(1000)
    values = compute_banzhaf_semivalues(
        utility,
        criterion,
        skip_converged=True,  # skip values that have converged (CAREFUL!)
    )
    ```
    This will compute the Banzhaf semivalues for `utility` until either the
    absolute standard error is below `1e-3` or `1000` updates have been
    performed. The `burn_in` parameter is used to discard the first `32` updates
    from the computation of the standard error. The `skip_converged` parameter
    is used to avoid computing more marginals for indices that have converged,
    which is useful if
    [AbsoluteStandardError][pydvl.value.stopping.AbsoluteStandardError] is met
    before [MaxUpdates][pydvl.value.stopping.MaxUpdates] for some indices.

!!! Warning
    Be careful not to reuse the same stopping criterion for different
    computations. The object has state and will not be reset between calls to
    value computation methods. If you need to reuse the same criterion, you
    should create a new instance.


## Creating stopping criteria

The easiest way is to declare a function implementing the interface
[StoppingCriterionCallable][pydvl.value.stopping.StoppingCriterionCallable] and
wrap it with [make_criterion()][pydvl.value.stopping.make_criterion]. This
creates a [StoppingCriterion][pydvl.value.stopping.StoppingCriterion] object
that can be composed with other stopping criteria.

Alternatively, and in particular if reporting of completion is required, one can
inherit from this class and implement the abstract methods `_check` and
[completion][pydvl.value.stopping.StoppingCriterion.completion].

## Combining stopping criteria

Objects of type [StoppingCriterion][pydvl.value.stopping.StoppingCriterion] can
be combined with the binary operators `&` (*and*), and `|` (*or*), following the
truth tables of [Status][pydvl.utils.status.Status]. The unary operator `~`
(*not*) is also supported. See
[StoppingCriterion][pydvl.value.stopping.StoppingCriterion] for details on how
these operations affect the behavior of the stopping criteria.


## References

[^1]: <a name="ghorbani_data_2019"></a>Ghorbani, A., Zou, J., 2019.
    [Data Shapley: Equitable Valuation of Data for Machine Learning](https://proceedings.mlr.press/v97/ghorbani19c.html).
    In: Proceedings of the 36th International Conference on Machine Learning, PMLR, pp. 2242–2251.
[^2]: <a name="wang_data_2023"></a>Wang, J.T. and Jia, R., 2023.
    [Data Banzhaf: A Robust Data Valuation Framework for Machine Learning](https://proceedings.mlr.press/v206/wang23e.html).
    In: Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, pp. 6388-6421.
"""

from __future__ import annotations

import abc
import logging
from time import time
from typing import Callable, Optional, Protocol, Type

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from pydvl.utils import Status
from pydvl.value import ValuationResult

__all__ = [
    "make_criterion",
    "AbsoluteStandardError",
    "StoppingCriterion",
    "MaxChecks",
    "MaxUpdates",
    "MinUpdates",
    "MaxTime",
    "HistoryDeviation",
    "RankCorrelation",
]

logger = logging.getLogger(__name__)


class StoppingCriterionCallable(Protocol):
    """Signature for a stopping criterion"""

    def __call__(self, result: ValuationResult) -> Status: ...


class StoppingCriterion(abc.ABC):
    """A composable callable object to determine whether a computation
    must stop.

    A `StoppingCriterion` is a callable taking a
    [ValuationResult][pydvl.value.result.ValuationResult] and returning a
    [Status][pydvl.value.result.Status]. It also keeps track of individual
    convergence of values with
    [converged][pydvl.value.stopping.StoppingCriterion.converged], and reports
    the overall completion of the computation with
    [completion][pydvl.value.stopping.StoppingCriterion.completion].

    Instances of `StoppingCriterion` can be composed with the binary operators
    `&` (*and*), and `|` (*or*), following the truth tables of
    [Status][pydvl.utils.status.Status]. The unary operator `~` (*not*) is
    also supported. These boolean operations act according to the following
    rules:

    - The results of `check()` are combined with the operator. See
      [Status][pydvl.utils.status.Status] for the truth tables.
    - The results of
      [converged][pydvl.value.stopping.StoppingCriterion.converged] are combined
      with the operator (returning another boolean array).
    - The [completion][pydvl.value.stopping.StoppingCriterion.completion]
      method returns the min, max, or the complement to 1 of the completions of
      the operands, for AND, OR and NOT respectively. This is required for cases
      where one of the criteria does not keep track of the convergence of single
      values, e.g. [MaxUpdates][pydvl.value.stopping.MaxUpdates], because
      [completion][pydvl.value.stopping.StoppingCriterion.completion] by
      default returns the mean of the boolean convergence array.

    # Subclassing

    Subclassing this class requires implementing a `check()` method that
    returns a [Status][pydvl.utils.status.Status] object based on a given
    [ValuationResult][pydvl.value.result.ValuationResult]. This method should
    update the attribute `_converged`, which is a boolean array indicating
    whether the value for each index has converged.
    When this does not make sense for a particular stopping criterion,
    [completion][pydvl.value.stopping.StoppingCriterion.completion] should be
    overridden to provide an overall completion value, since its default
    implementation attempts to compute the mean of `_converged`.

    Args:
        modify_result: If `True` the status of the input
            [ValuationResult][pydvl.value.result.ValuationResult] is modified in
            place after the call.
    """

    _converged: NDArray[
        np.bool_
    ]  #: A boolean array indicating whether the corresponding element has converged

    def __init__(self, modify_result: bool = True):
        self.modify_result = modify_result
        self._converged = np.full(0, False)

    @abc.abstractmethod
    def _check(self, result: ValuationResult) -> Status:
        """Check whether the computation should stop."""
        ...

    def completion(self) -> float:
        """Returns a value between 0 and 1 indicating the completion of the
        computation.
        """
        if self.converged.size == 0:
            return 0.0
        return float(np.mean(self.converged).item())

    def reset(self):
        pass

    @property
    def converged(self) -> NDArray[np.bool_]:
        """Returns a boolean array indicating whether the values have converged
        for each data point.

        Inheriting classes must set the `_converged` attribute in their
        `check()`.

        Returns:
            A boolean array indicating whether the values have converged for
            each data point.
        """
        return self._converged

    def __str__(self):
        return type(self).__name__

    def __call__(self, result: ValuationResult) -> Status:
        """Calls `check()`, maybe updating the result."""
        if len(result) == 0:
            logger.warning(
                "At least one iteration finished but no results where generated. "
                "Please check that your scorer and utility return valid numbers."
            )
        status = self._check(result)
        if self.modify_result:  # FIXME: this is not nice
            result._status = status
        return status

    def __and__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        return make_criterion(
            fun=lambda result: self._check(result) & other._check(result),
            converged=lambda: self.converged & other.converged,
            completion=lambda: min(self.completion(), other.completion()),
            name=f"Composite StoppingCriterion: {str(self)} AND {str(other)}",
        )(modify_result=self.modify_result or other.modify_result)

    def __or__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        return make_criterion(
            fun=lambda result: self._check(result) | other._check(result),
            converged=lambda: self.converged | other.converged,
            completion=lambda: max(self.completion(), other.completion()),
            name=f"Composite StoppingCriterion: {str(self)} OR {str(other)}",
        )(modify_result=self.modify_result or other.modify_result)

    def __invert__(self) -> "StoppingCriterion":
        return make_criterion(
            fun=lambda result: ~self._check(result),
            converged=lambda: ~self.converged,
            completion=lambda: 1 - self.completion(),
            name=f"Composite StoppingCriterion: NOT {str(self)}",
        )(modify_result=self.modify_result)


def make_criterion(
    fun: StoppingCriterionCallable,
    converged: Callable[[], NDArray[np.bool_]] | None = None,
    completion: Callable[[], float] | None = None,
    name: str | None = None,
) -> Type[StoppingCriterion]:
    """Create a new [StoppingCriterion][pydvl.value.stopping.StoppingCriterion] from a function.
    Use this to enable simpler functions to be composed with bitwise operators

    Args:
        fun: The callable to wrap.
        converged: A callable that returns a boolean array indicating what
            values have converged.
        completion: A callable that returns a value between 0 and 1 indicating
            the rate of completion of the computation. If not provided, the fraction
            of converged values is used.
        name: The name of the new criterion. If `None`, the `__name__` of
            the function is used.

    Returns:
        A new subclass of [StoppingCriterion][pydvl.value.stopping.StoppingCriterion].
    """

    class WrappedCriterion(StoppingCriterion):
        def __init__(self, modify_result: bool = True):
            super().__init__(modify_result=modify_result)
            self._name = name or getattr(fun, "__name__", "WrappedCriterion")

        def _check(self, result: ValuationResult) -> Status:
            return fun(result)

        @property
        def converged(self) -> NDArray[np.bool_]:
            if converged is None:
                return super().converged
            return converged()

        def __str__(self):
            return self._name

        def completion(self) -> float:
            if completion is None:
                return super().completion()
            return completion()

    return WrappedCriterion


class AbsoluteStandardError(StoppingCriterion):
    r"""Determine convergence based on the standard error of the values.

    If $s_i$ is the standard error for datum $i$, then this criterion returns
    [Converged][pydvl.utils.status.Status] if $s_i < \epsilon$ for all $i$ and a
    threshold value $\epsilon \gt 0$.

    Args:
        threshold: A value is considered to have converged if the standard
            error is below this threshold. A way of choosing it is to pick some
            percentage of the range of the values. For Shapley values this is
            the difference between the maximum and minimum of the utility
            function (to see this substitute the maximum and minimum values of
            the utility into the marginal contribution formula).
        fraction: The fraction of values that must have converged for the
            criterion to return [Converged][pydvl.utils.status.Status].
        burn_in: The number of iterations to ignore before checking for
            convergence. This is required because computations typically start
            with zero variance, as a result of using
            [zeros()][pydvl.value.result.ValuationResult.zeros]. The default is
            set to an arbitrary minimum which is usually enough but may need to
            be increased.
    """

    def __init__(
        self,
        threshold: float,
        fraction: float = 1.0,
        burn_in: int = 4,
        modify_result: bool = True,
    ):
        super().__init__(modify_result=modify_result)
        self.threshold = threshold
        self.fraction = fraction
        self.burn_in = burn_in

    def _check(self, result: ValuationResult) -> Status:
        self._converged = (result.stderr < self.threshold) & (
            result.counts > self.burn_in
        )
        if np.mean(self._converged) >= self.fraction:
            return Status.Converged
        return Status.Pending

    def __str__(self):
        return f"AbsoluteStandardError(threshold={self.threshold}, fraction={self.fraction}, burn_in={self.burn_in})"


class MaxChecks(StoppingCriterion):
    """Terminate as soon as the number of checks exceeds the threshold.

    A "check" is one call to the criterion.

    Args:
        n_checks: Threshold: if `None`, no _check is performed,
            effectively creating a (never) stopping criterion that always returns
            `Pending`.
    """

    def __init__(self, n_checks: Optional[int], modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        if n_checks is not None and n_checks < 1:
            raise ValueError("n_iterations must be at least 1 or None")
        self.n_checks = n_checks
        self._count = 0

    def _check(self, result: ValuationResult) -> Status:
        if self.n_checks:
            self._count += 1
            if self._count >= self.n_checks:
                self._converged = np.ones_like(result.values, dtype=bool)
                return Status.Converged
        return Status.Pending

    def completion(self) -> float:
        if self.n_checks:
            return min(1.0, self._count / self.n_checks)
        return 0.0

    def reset(self):
        self._count = 0

    def __str__(self):
        return f"MaxChecks(n_checks={self.n_checks})"


class MaxUpdates(StoppingCriterion):
    """Terminate if any number of value updates exceeds or equals the given
    threshold.

    !!! Note
        If you want to ensure that **all** values have been updated, you
        probably want [MinUpdates][pydvl.value.stopping.MinUpdates] instead.

    This checks the `counts` field of a
    [ValuationResult][pydvl.value.result.ValuationResult], i.e. the number of
    times that each index has been updated. For powerset samplers, the maximum
    of this number coincides with the maximum number of subsets sampled. For
    permutation samplers, it coincides with the number of permutations sampled.

    Args:
        n_updates: Threshold: if `None`, no _check is performed,
            effectively creating a (never) stopping criterion that always returns
            `Pending`.
    """

    def __init__(self, n_updates: Optional[int], modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        if n_updates is not None and n_updates < 1:
            raise ValueError("n_updates must be at least 1 or None")
        self.n_updates = n_updates
        self.last_max = 0

    def _check(self, result: ValuationResult) -> Status:
        if self.n_updates:
            self._converged = result.counts >= self.n_updates
            try:
                self.last_max = int(np.max(result.counts))
                if self.last_max >= self.n_updates:
                    return Status.Converged
            except ValueError:  # empty counts array. This should not happen
                pass
        return Status.Pending

    def completion(self) -> float:
        if self.n_updates:
            return self.last_max / self.n_updates
        return 0.0

    def __str__(self):
        return f"MaxUpdates(n_updates={self.n_updates})"


class MinUpdates(StoppingCriterion):
    """Terminate as soon as all value updates exceed or equal the given threshold.

    This checks the `counts` field of a
    [ValuationResult][pydvl.value.result.ValuationResult], i.e. the number of times that
    each index has been updated. For powerset samplers, the minimum of this
    number is a lower bound for the number of subsets sampled. For
    permutation samplers, it lower-bounds the amount of permutations sampled.

    Args:
        n_updates: Threshold: if `None`, no _check is performed,
            effectively creating a (never) stopping criterion that always returns
            `Pending`.
    """

    def __init__(self, n_updates: Optional[int], modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        self.n_updates = n_updates
        self.last_min = 0

    def _check(self, result: ValuationResult) -> Status:
        if self.n_updates is not None:
            self._converged = result.counts >= self.n_updates
            try:
                self.last_min = int(np.min(result.counts))
                if self.last_min >= self.n_updates:
                    return Status.Converged
            except ValueError:  # empty counts array. This should not happen
                pass
        return Status.Pending

    def completion(self) -> float:
        if self.n_updates:
            return self.last_min / self.n_updates
        return 0.0

    def __str__(self):
        return f"MinUpdates(n_updates={self.n_updates})"


class MaxTime(StoppingCriterion):
    """Terminate if the computation time exceeds the given number of seconds.

    Checks the elapsed time since construction

    Args:
        seconds: Threshold: The computation is terminated if the elapsed time
            between object construction and a _check exceeds this value. If `None`,
            no _check is performed, effectively creating a (never) stopping criterion
            that always returns `Pending`.
    """

    def __init__(self, seconds: Optional[float], modify_result: bool = True):
        super().__init__(modify_result=modify_result)
        self.max_seconds = seconds or np.inf
        if self.max_seconds <= 0:
            raise ValueError("Number of seconds for MaxTime must be positive or None")
        self.start = time()

    def _check(self, result: ValuationResult) -> Status:
        if self._converged is None:
            self._converged = np.full(result.values.shape, False)
        if time() > self.start + self.max_seconds:
            self._converged.fill(True)
            return Status.Converged
        return Status.Pending

    def completion(self) -> float:
        if self.max_seconds is None:
            return 0.0
        return (time() - self.start) / self.max_seconds

    def reset(self):
        self.start = time()

    def __str__(self):
        return f"MaxTime(seconds={self.max_seconds})"


class HistoryDeviation(StoppingCriterion):
    r"""A simple check for relative distance to a previous step in the
    computation.

    The method used by (Ghorbani and Zou, 2019)<sup><a href="#ghorbani_data_2019">1</a></sup> computes the relative
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
    pinned to that state. Once all indices have converged the method has
    converged.

    !!! Warning
        This criterion is meant for the reproduction of the results in the paper,
        but we do not recommend using it in practice.

    Args:
        n_steps: Checkpoint values every so many updates and use these saved
            values to compare.
        rtol: Relative tolerance for convergence ($\epsilon$ in the formula).
        pin_converged: If `True`, once an index has converged, it is pinned
    """

    _memory: NDArray[np.float64]

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
        self._memory = None  # type: ignore

    def _check(self, r: ValuationResult) -> Status:
        if self._memory is None:
            self._memory = np.full((len(r.values), self.n_steps + 1), np.inf)
            self._converged = np.full(len(r), False)
            return Status.Pending

        # shift left: last column is the last set of values
        self._memory = np.concatenate(
            [self._memory[:, 1:], r.values.reshape(-1, 1)], axis=1
        )

        # Look at indices that have been updated more than n_steps times
        ii = np.where(r.counts > self.n_steps)
        if len(ii) > 0:
            curr = self._memory[:, -1]
            saved = self._memory[:, 0]
            diffs = np.abs(curr[ii] - saved[ii])
            quots = np.divide(diffs, curr[ii], out=diffs, where=curr[ii] != 0)
            # quots holds the quotients when the denominator is non-zero, and
            # the absolute difference, which is just the memory, otherwise.
            if len(quots) > 0 and np.mean(quots) < self.rtol:
                self._converged = self.update_op(
                    self._converged, r.counts > self.n_steps
                )  # type: ignore
                if np.all(self._converged):
                    return Status.Converged
        return Status.Pending

    def reset(self):
        self._memory = None  # type: ignore

    def __str__(self):
        return f"HistoryDeviation(n_steps={self.n_steps}, rtol={self.rtol})"


class RankCorrelation(StoppingCriterion):
    r"""A check for stability of Spearman correlation between checks.

    When the change in rank correlation between two successive iterations is
    below a given threshold, the computation is terminated.
    The criterion computes the Spearman correlation between two successive iterations.
    The Spearman correlation uses the ordering indices of the given values and
    correlates them. This means it focuses on the order of the elements instead of their
    exact values. If the order stops changing (meaning the Banzhaf semivalues estimates
    converge), the criterion stops the algorithm.

    This criterion is used in (Wang et. al.)<sup><a href="wang_data_2023">2</a></sup>.

    Args:
        rtol: Relative tolerance for convergence ($\epsilon$ in the formula)
        modify_result: If `True`, the status of the input
            [ValuationResult][pydvl.value.result.ValuationResult] is modified in
            place after the call.
        burn_in: The minimum number of iterations before checking for
            convergence. This is required because the first correlation is
            meaningless.

    !!! tip "Added in 0.9.0"
    """

    def __init__(
        self,
        rtol: float,
        burn_in: int,
        modify_result: bool = True,
    ):
        super().__init__(modify_result=modify_result)
        if rtol <= 0 or rtol >= 1:
            raise ValueError("rtol must be in (0, 1)")
        self.rtol = rtol
        self.burn_in = burn_in
        self._memory: NDArray[np.float64] | None = None
        self._corr = 0.0
        self._completion = 0.0
        self._iterations = 0

    def _check(self, r: ValuationResult) -> Status:
        self._iterations += 1
        if self._memory is None:
            self._memory = r.values.copy()
            self._converged = np.full(len(r), False)
            return Status.Pending

        corr = spearmanr(self._memory, r.values)[0]
        self._memory = r.values.copy()
        self._update_completion(corr)
        if (
            np.isclose(corr, self._corr, rtol=self.rtol)
            and self._iterations > self.burn_in
        ):
            self._converged = np.full(len(r), True)
            logger.debug(
                f"RankCorrelation has converged with {corr=} in iteration {self._iterations}"
            )
            return Status.Converged
        self._corr = np.nan_to_num(corr, nan=0.0)
        return Status.Pending

    def _update_completion(self, corr: float) -> None:
        if np.isnan(corr):
            self._completion = 0.0
        elif not np.isclose(corr, self._corr, rtol=self.rtol):
            self._completion = corr
            # self.rtol / np.abs(corr - self._corr) might be another option
        else:
            self._completion = 1.0

    def completion(self) -> float:
        return self._completion

    def reset(self):
        self._memory = None  # type: ignore
        self._corr = 0.0

    def __str__(self):
        return f"RankCorrelation(rtol={self.rtol})"

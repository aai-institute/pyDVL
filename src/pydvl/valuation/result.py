"""
This module collects types and methods for the inspection of the results of
valuation algorithms.

The most important class is [ValuationResult][pydvl.valuation.result.ValuationResult],
which provides access to raw values, as well as convenient behaviour as a `Sequence`
with extended indexing and updating abilities, and conversion to [pandas
DataFrames][pandas.DataFrame].

## Indexing and slicing

Indexing and slicing of results is supported in a natural way and
[ValuationResult][pydvl.valuation.result.ValuationResult] objects are returned. Indexing
follows the sorting order. See the class documentation for more on this.

Setting items and slices is also possible with other valuation results. Index and name
clashes are detected and raise an exception. Note that any sorted state is potentially
lost when setting items or slices.

## Addition

Results can be added together with the standard `+` operator. Because values
are typically running averages of iterative algorithms, addition behaves like a
weighted average of the two results, with the weights being the number of
updates in each result: adding two results is the same as generating one result
with the mean of the values of the two results as values. The variances are
updated accordingly. See [ValuationResult][pydvl.valuation.result.ValuationResult] for
details.

## Comparing

Results can be compared with the equality operator. The comparison is "semantic" in the
sense that it's the valuation for data indices that matters and not the order in which
they are in the `ValuationResult`. Values, variances and counts are compared.

## Sorting

Results can also be sorted **in place** by value, variance or number of updates, see
[sort()][pydvl.valuation.result.ValuationResult.sort]. All the properties
[ValuationResult.values][pydvl.valuation.result.ValuationResult.values],
[ValuationResult.variances][pydvl.valuation.result.ValuationResult.variances],
[ValuationResult.counts][pydvl.valuation.result.ValuationResult.counts],
[ValuationResult.indices][pydvl.valuation.result.ValuationResult.indices],
[ValuationResult.stderr][pydvl.valuation.result.ValuationResult.stderr],
[ValuationResult.names][pydvl.valuation.result.ValuationResult.names]
are then sorted according to the same order.

## Updating

Updating results as new values arrive from workers in valuation algorithms can depend on
the algorithm used. The most common case is to use the
[LogResultUpdater][pydvl.valuation.result.LogResultUpdater] class, which uses the
log-sum-exp trick to update the values and variances for better numerical stability.
This is the default behaviour with the base
[IndexSampler][pydvl.valuation.samplers.base.IndexSampler], but other sampling schemes
might require different ones. In particular,
[MSRResultUpdater][pydvl.valuation.samplers.msr.MSRResultUpdater] must keep track of
separate positive and negative updates.


## Factories

Besides [copy()][pydvl.valuation.result.ValuationResult.copy],the most commonly used
factory method is
[ValuationResult.zeros()][pydvl.valuation.result.ValuationResult.zeros], which
creates a result object with all values, variances and counts set to zero.

[ValuationResult.empty()][pydvl.valuation.result.ValuationResult.empty] creates an
empty result object, which can be used as a starting point for adding results
together. **Any metadata in empty results is discarded when added to other results.**

Finally,
[ValuationResult.from_random()][pydvl.valuation.result.ValuationResult.from_random]
samples random values uniformly.
"""

from __future__ import annotations

import collections.abc
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import total_ordering
from numbers import Integral
from typing import (
    Any,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    Union,
    cast,
    overload,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import Self

from pydvl.utils import log_running_moments
from pydvl.utils.status import Status
from pydvl.utils.types import Seed
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.types import IndexSetT, IndexT, NameT, ValueUpdate, ValueUpdateT

__all__ = ["LogResultUpdater", "ResultUpdater", "ValuationResult", "ValueItem"]

logger = logging.getLogger(__name__)


@total_ordering
@dataclass
class ValueItem:
    """The result of a value computation for one datum.

    `ValueItems` can be compared with the usual operators, forming a total
    order. Comparisons take only the `idx`, `name` and `value` into account.

    !!! todo
        Maybe have a mode of comparison taking the `variance` into account.

    Attributes:
        idx: Index of the sample with this value in the original
            [Dataset][pydvl.utils.dataset.Dataset]
        name: Name of the sample if it was provided. Otherwise, `str(idx)`
        value: The value
        variance: Variance of the marginals from which the value was computed.
        count: Number of updates for this value
    """

    idx: IndexT
    name: NameT
    value: float
    variance: float | None
    count: int | None

    def _comparable(self, other: object) -> bool:
        if not isinstance(other, ValueItem):
            raise TypeError(f"Cannot compare ValueItem with {type(other)}")
        return bool(self.idx == other.idx and self.name == other.name)

    def __lt__(self, other: object) -> bool:
        return self._comparable(other) and self.value < other.value  # type: ignore

    def __le__(self, other: object) -> bool:
        return self._comparable(other) and self.value <= other.value  # type: ignore

    def __ge__(self, other: object) -> bool:
        return self._comparable(other) and self.value >= other.value  # type: ignore

    def __gt__(self, other: object) -> bool:
        return not self.__lt__(other)

    def __eq__(self, other: object) -> bool:
        return self._comparable(other) and bool(np.isclose(self.value, other.value))  # type: ignore

    def __index__(self) -> IndexT:
        return self.idx


class ValuationResult(collections.abc.Sequence, Iterable[ValueItem]):
    """Objects of this class hold the results of valuation algorithms.

    These include indices in the original [Dataset][pydvl.utils.dataset.Dataset],
    any data names (e.g. group names in [GroupedDataset][pydvl.utils.dataset.GroupedDataset]),
    the values themselves, and variance of the computation in the case of Monte
    Carlo methods. `ValuationResults` can be iterated over like any `Sequence`:
    `iter(valuation_result)` returns a generator of
    [ValueItem][pydvl.valuation.result.ValueItem] in the order in which the object
    is sorted.

    ## Indexing

    Indexing is sort-based, when accessing any of the attributes
    [values][pydvl.valuation.result.ValuationResult.values],
    [variances][pydvl.valuation.result.ValuationResult.variances],
    [counts][pydvl.valuation.result.ValuationResult.counts] and
    [indices][pydvl.valuation.result.ValuationResult.indices], as well as when iterating
    over the object, or using the item access operator, both getter and setter. The
    "position" is either the original sequence in which the data was passed to the
    constructor, or the sequence in which the object has been sorted, see below.
    One can retrieve the sorted position for a given data index using the method
    [positions()][pydvl.valuation.result.ValuationResult.positions].

    Some methods use data indices instead. This is the case for
    [get()][pydvl.valuation.result.ValuationResult.get].

    ## Sorting

    Results can be sorted in-place with
    [sort()][pydvl.valuation.result.ValuationResult.sort], or alternatively using
    python's standard `sorted()` and `reversed()` Note that sorting values affects how
    iterators and the object itself as `Sequence` behave: `values[0]` returns a
    [ValueItem][pydvl.valuation.result.ValueItem] with the highest or lowest ranking
    point if this object is sorted by descending or ascending value, respectively.the methods If
    unsorted, `values[0]` returns the `ValueItem` at position 0, which has data index
    `indices[0]` in the [Dataset][pydvl.utils.dataset.Dataset].

    The same applies to direct indexing of the `ValuationResult`: the index
    is positional, according to the sorting. It does not refer to the "data
    index". To sort according to data index, use
    [sort()][pydvl.valuation.result.ValuationResult.sort] with `key="index"`.

    In order to access [ValueItem][pydvl.valuation.result.ValueItem] objects by their
    data index, use [get()][pydvl.valuation.result.ValuationResult.get], or use
    [positions()][pydvl.valuation.result.ValuationResult.positions] to convert data
    indices to positions.

    !!! tip "Converting back and forth from data indices and positions"
        `data_indices = result.indices[result.positions(data_indices)]` is a noop.

    ## Operating on results

    Results can be added to each other with the `+` operator. Means and variances
    are correctly updated accordingly using the Welford algorithm.

    Empty objects behave in a special way, see
    [empty()][pydvl.valuation.result.ValuationResult.empty].

    Args:
        values: An array of values. If omitted, defaults to an empty array
            or to an array of zeros if `indices` are given.
        indices: An optional array of indices in the original dataset. If
            omitted, defaults to `np.arange(len(values))`. **Warning:** It is
            common to pass the indices of a [Dataset][pydvl.utils.dataset.Dataset]
            here. Attention must be paid in a parallel context to copy them to
            the local process. Just do `indices=np.copy(data.indices)`.
        variances: An optional array of variances of the marginals from which the values
            are computed.
        counts: An optional array with the number of updates for each value.
            Defaults to an array of ones.
        data_names: Names for the data points. Defaults to index numbers if not set.
        algorithm: The method used.
        status: The end status of the algorithm.
        sort: Whether to sort the indices. Defaults to `None` for no sorting. Set to
            `True` for ascending order by value, `False` for descending. See above how
            sorting affects usage as an iterable or sequence.
        extra_values: Additional values that can be passed as keyword arguments.
            This can contain, for example, the least core value.

    Raises:
         ValueError: If input arrays have mismatching lengths.

    ??? tip "Changed in 0.10.0"
        Changed the behaviour of sorting, slicing, and indexing.
    """

    _data_indices: NDArray[IndexT]
    _values: NDArray[np.float64]
    _counts: NDArray[np.int_]
    _variances: NDArray[np.float64]
    _data: Dataset
    _names: NDArray[NameT]
    _algorithm: str
    _status: Status
    # None for unsorted, True for ascending, False for descending
    _sort_order: bool | None
    _extra_values: dict[str, Any]
    _positions_to_indices: NDArray[IndexT]
    _indices_to_positions: NDArray[IndexT]

    __version__: str = "1.0"

    def __init__(
        self,
        *,
        values: Sequence[np.float64] | NDArray[np.float64],
        variances: Sequence[np.float64] | NDArray[np.float64] | None = None,
        counts: Sequence[np.int_] | NDArray[np.int_] | None = None,
        indices: Sequence[IndexT] | NDArray[IndexT] | None = None,
        data_names: Sequence[NameT] | NDArray[NameT] | None = None,
        algorithm: str = "",
        status: Status = Status.Pending,
        sort: bool | None = None,
        **extra_values: Any,
    ):
        if variances is not None and len(variances) != len(values):
            raise ValueError(
                f"Lengths of values ({len(values)}) "
                f"and variances ({len(variances)}) do not match"
            )
        if data_names is not None and len(data_names) != len(values):
            raise ValueError(
                f"Lengths of values ({len(values)}) "
                f"and data_names ({len(data_names)}) do not match"
            )
        if indices is not None and len(indices) != len(values):
            raise ValueError(
                f"Lengths of values ({len(values)}) "
                f"and indices ({len(indices)}) do not match"
            )

        self._algorithm = algorithm
        self._status = Status(status)  # Just in case we are given a string
        self._values = np.asarray(values, dtype=np.float64)
        self._variances = (
            np.zeros_like(values) if variances is None else np.asarray(variances)
        )
        self._counts = (
            np.ones_like(values, dtype=int) if counts is None else np.asarray(counts)
        )
        self._sort_order = None
        self._extra_values = extra_values or {}

        # Internal indices -> data indices
        self._data_indices = self._create_indices_array(indices, len(self._values))
        self._names = self._create_names_array(data_names, self._data_indices)

        # Data indices -> Internal indices
        self._indices = {idx: pos for pos, idx in enumerate(self._data_indices)}

        # Sorted indices ("positions") -> Internal indices
        self._positions_to_indices = np.arange(len(self._values), dtype=np.int_)

        # Internal indices -> Sorted indices ("positions")
        self._indices_to_positions = np.arange(len(self._values), dtype=np.int_)

        if sort is not None:
            self.sort(reverse=not sort)

    def sort(
        self,
        reverse: bool = False,
        # Need a "Comparable" type here
        key: Literal["value", "variance", "index", "name"] = "value",
    ) -> None:
        """Sorts the indices **in place** in ascending order by `key`.

        Once sorted, iteration over the results, and indexing of all the
        properties
        [ValuationResult.values][pydvl.valuation.result.ValuationResult.values],
        [ValuationResult.variances][pydvl.valuation.result.ValuationResult.variances],
        [ValuationResult.stderr][pydvl.valuation.result.ValuationResult.stderr],
        [ValuationResult.counts][pydvl.valuation.result.ValuationResult.counts],
        [ValuationResult.indices][pydvl.valuation.result.ValuationResult.indices]
        and [ValuationResult.names][pydvl.valuation.result.ValuationResult.names]
        will follow the same order.

        Args:
            reverse: Whether to sort in descending order.
            key: The key to sort by. Defaults to
                [ValueItem.value][pydvl.valuation.result.ValueItem].
        """
        keymap = {
            "index": "_data_indices",
            "value": "_values",
            "variance": "_variances",
            "name": "_names",
        }
        self._positions_to_indices = np.argsort(getattr(self, keymap[key])).astype(int)
        if reverse:
            self._positions_to_indices = self._positions_to_indices[::-1]
        self._sort_order = not reverse
        self._indices_to_positions = np.argsort(self._positions_to_indices).astype(int)

    def positions(self, data_indices: IndexSetT | list[IndexT]) -> IndexSetT:
        """Return the location (indices) within the `ValuationResult` for the given
        data indices.

        Sorting is taken into account. This operation is the inverse of indexing the
        [indices][pydvl.valuation.result.ValuationResult.indices] property:

        ```python
        np.all(v.indices[v.positions(data_indices)] == data_indices) == True
        ```
        """
        indices = [self._indices[idx] for idx in data_indices]
        return self._indices_to_positions[indices]

    @property
    def values(self) -> NDArray[np.float64]:
        """The values, possibly sorted."""
        return self._values[self._positions_to_indices]

    @property
    def variances(self) -> NDArray[np.float64]:
        """Variances of the marginals from which values were computed, possibly sorted.

        Note that this is not the variance of the value estimate, but the sample
        variance of the marginals used to compute it.

        """
        return self._variances[self._positions_to_indices]

    @property
    def stderr(self) -> NDArray[np.float64]:
        """Standard errors of the value estimates, possibly sorted."""
        return cast(
            NDArray[np.float64], np.sqrt(self.variances / np.maximum(1, self.counts))
        )

    @property
    def counts(self) -> NDArray[np.int_]:
        """The raw counts, possibly sorted."""
        return self._counts[self._positions_to_indices]

    @property
    def indices(self) -> NDArray[IndexT]:
        """The indices for the values, possibly sorted.

        If the object is unsorted, then these are the same as declared at
        construction or `np.arange(len(values))` if none were passed.
        """
        return self._data_indices[self._positions_to_indices]

    @property
    def names(self) -> NDArray[NameT]:
        """The names for the values, possibly sorted.
        If the object is unsorted, then these are the same as declared at
        construction or `np.arange(len(values))` if none were passed.
        """
        return self._names[self._positions_to_indices]

    @property
    def status(self) -> Status:
        return self._status

    @property
    def algorithm(self) -> str:
        return self._algorithm

    def copy(self) -> ValuationResult:
        """Returns a copy of the object."""
        return ValuationResult(
            values=self._values.copy(),
            variances=self._variances.copy(),
            counts=self._counts.copy(),
            indices=self._data_indices.copy(),
            data_names=self._names.copy(),
            algorithm=self._algorithm,
            status=self._status,
            sort=self._sort_order,
            **self._extra_values,
        )

    def __getattr__(self, attr: str) -> Any:
        """Allows access to extra values as if they were properties of the instance."""
        # This is here to avoid a RecursionError when copying or pickling the object
        if attr == "_extra_values":
            if "_extra_values" in self.__dict__:
                return self.__dict__["_extra_values"]
            return {}  # Return empty dict as fallback to prevent pickle from failing
        try:
            return self._extra_values[attr]
        except KeyError as e:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {attr}"
            ) from e

    def _key_to_positions(self, key: Union[slice, Iterable[int], int]) -> list[int]:
        if isinstance(key, slice):
            return [i for i in range(*key.indices(len(self)))]

        if isinstance(key, collections.abc.Sequence) and not isinstance(
            key, (str, bytes)
        ):
            try:
                return [int(k) for k in key]
            except TypeError as e:
                raise TypeError(
                    f"Indices must be integers, sequences or slices. {key=} has type {type(key)}"
                ) from e
        if isinstance(key, np.ndarray) and np.issubdtype(key.dtype, np.integer):
            return cast(list[int], key.astype(int).tolist())
        if isinstance(key, Integral):
            idx = int(key)
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range (0, {len(self)}).")
            return [idx]

        raise TypeError(
            f"Indices must be integers, integer sequences or ndarrays, or slices. "
            f"{key=} has type {type(key)}"
        )

    @overload
    def __getitem__(self, key: int) -> ValuationResult: ...

    @overload
    def __getitem__(self, key: slice) -> ValuationResult: ...

    @overload
    def __getitem__(self, key: Iterable[int]) -> ValuationResult: ...

    def __getitem__(self, key: Union[slice, Iterable[int], int]) -> ValuationResult:
        """Get a ValuationResult for the given key.

        The key can be an integer, a slice, or an iterable of integers.
        The returned object is a new `ValuationResult` with all metadata copied, except
        for the sorting order. If the key is a slice or sequence, the returned object
        will contain the items **in the order specified by the sequence**.

        Returns:
            A new object containing only the selected items.
        """

        positions = self._key_to_positions(key)

        # Convert positions to original indices in the sort order
        sort_indices = self._positions_to_indices[positions]

        return ValuationResult(
            values=self._values[sort_indices].copy(),
            variances=self._variances[sort_indices].copy(),
            counts=self._counts[sort_indices].copy(),
            indices=self._data_indices[sort_indices].copy(),
            data_names=self._names[sort_indices].copy(),
            algorithm=self._algorithm,
            status=self._status,
            # sort=self._sort_order,  # makes no sense
            **self._extra_values,
        )

    @overload
    def __setitem__(self, key: int, value: ValuationResult) -> None: ...

    @overload
    def __setitem__(self, key: slice, value: ValuationResult) -> None: ...

    @overload
    def __setitem__(self, key: Iterable[int], value: ValuationResult) -> None: ...

    def __setitem__(
        self, key: Union[slice, Iterable[int], int], value: ValuationResult
    ) -> None:
        """Set items in the `ValuationResult` using another `ValuationResult`.

        This method provides a symmetrical counterpart to `__getitem__`, both
        operating on `ValuationResult` objects.

        The key can be an integer, a slice, or an iterable of integers.
        The value must be a `ValuationResult` with length matching the number of
        positions specified by key.

        Args:
            key: Position(s) to set
            value: A ValuationResult to set at the specified position(s)

        Raises:
            TypeError: If value is not a ValuationResult
            ValueError: If value's length doesn't match the number of positions
                specified by the key
        """
        if not isinstance(value, ValuationResult):
            raise TypeError(
                f"Value must be a ValuationResult, got {type(value)}. "
                f"To set individual ValueItems, use the set() method instead."
            )

        positions = self._key_to_positions(key)

        if len(value) != len(positions):
            raise ValueError(
                f"Cannot set {len(positions)} positions with a ValuationResult of length {len(value)}"
            )

        # Convert sorted positions (user-facing) to original indices in the sort order
        destination = self._positions_to_indices[positions]
        # For the source, use the first sorted n items
        source = list(range(len(positions)))

        # Check that the operation won't result in duplicate indices or names
        new_indices = self._data_indices.copy()
        new_indices[destination] = value.indices[source]
        new_names = self._names.copy()
        new_names[destination] = value.names[source]

        if len(np.unique(new_indices)) != len(new_indices):
            raise ValueError("Operation would result in duplicate indices")
        if len(np.unique(new_names)) != len(new_names):
            raise ValueError("Operation would result in duplicate names")

        # Update data index -> internal index mapping
        for data_idx in self._data_indices[destination]:
            del self._indices[data_idx]
        for data_idx, dest in zip(value.indices[source], destination):
            self._indices[data_idx] = dest

        self._data_indices[destination] = value.indices[source]
        self._names[destination] = value.names[source]
        self._values[destination] = value.values[source]
        self._variances[destination] = value.variances[source]
        self._counts[destination] = value.counts[source]

    def set(self, data_idx: IndexT, value: ValueItem) -> Self:
        """Set a [ValueItem][pydvl.valuation.result.ValueItem] in the result by its data
        index.

        This is the complement to the [get()][pydvl.valuation.result.ValuationResult.get]
        method and allows setting individual `ValueItems` directly by their data index
        rather than (sort-) position.

        Args:
            data_idx: Data index of the value to set
            value: The data to set

        Returns:
            A reference to self for method chaining

        Raises:
            IndexError: If the index is not found
            ValueError: If the `ValueItem`'s idx doesn't match `data_idx`
        """
        if value.idx != data_idx:
            raise ValueError(
                f"ValueItem's idx ({value.idx}) doesn't match the provided data_idx ({data_idx})"
            )

        try:
            pos = self._indices[data_idx]
        except KeyError:
            raise IndexError(f"Index {data_idx} not found in ValuationResult")

        self._data_indices[pos] = value.idx
        self._names[pos] = value.name
        self._values[pos] = value.value
        self._variances[pos] = value.variance
        self._counts[pos] = value.count

        return self

    def __iter__(self) -> Iterator[ValueItem]:
        """Iterate over the results returning [ValueItem][pydvl.valuation.result.ValueItem] objects.
        To sort in place before iteration, use [sort()][pydvl.valuation.result.ValuationResult.sort].
        """
        for pos in self._positions_to_indices:
            yield ValueItem(
                self._data_indices[pos],
                self._names[pos],
                self._values[pos],
                self._variances[pos],
                self._counts[pos],
            )

    def __len__(self):
        return len(self._data_indices)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ValuationResult):
            raise TypeError(f"Cannot compare ValuationResult with {type(other)}")

        if not np.array_equal(
            np.sort(self._data_indices), np.sort(other._data_indices)
        ):
            return False

        if self._algorithm != other._algorithm or self._status != other._status:
            return False

        if set(self._extra_values.keys()) != set(other._extra_values.keys()):
            return False

        for k, v in self._extra_values.items():
            if k not in other._extra_values:
                return False
            if isinstance(v, np.ndarray):
                if not np.array_equal(v, other._extra_values[k], equal_nan=True):
                    return False
            else:
                try:
                    if np.isnan(v) and np.isnan(other._extra_values[k]):
                        continue
                except TypeError:
                    if v != other._extra_values[k]:
                        return False

        self_pos = self.positions(self._data_indices)
        other_pos = other.positions(other._data_indices)

        if not (
            np.array_equal(
                self._values[self_pos], other._values[other_pos], equal_nan=True
            )
            and np.array_equal(
                self._variances[self_pos], other._variances[other_pos], equal_nan=True
            )
            and np.array_equal(self._counts[self_pos], other._counts[other_pos])
            and np.array_equal(self._names[self_pos], other._names[other_pos])
        ):
            return False

        return True

    def __str__(self) -> str:
        repr_string = (
            f"{self.__class__.__name__}("
            f"algorithm='{self._algorithm}',"
            f"status='{self._status.value}',"
            f"values={np.array_str(self.values, precision=4, suppress_small=True)},"
            f"indices={np.array_str(self.indices)},"
            f"names={np.array_str(self.names)},"
            f"counts={np.array_str(self.counts)}"
        )
        for k, v in self._extra_values.items():
            repr_string += f", {k}={v}"
        repr_string += ")"
        return repr_string

    def __getstate__(self):
        self.__dict__["_class_version"] = self.__class__.__version__
        return self.__dict__

    def __setstate__(self, state):
        # Lazy future proofing: allow user-defined upgrade hook if provided
        if (upgrade := getattr(self.__class__, "__upgrade_state__", None)) is not None:
            state = upgrade(state)
        if state.get("_class_version") != (expected := self.__class__.__version__):
            raise ValueError(
                f"Pickled ValuationResult version mismatch: expected {expected}, "
                f"got {state.get('_class_version')}"
            )
        self.__dict__ = state

    def _check_compatible(self, other: ValuationResult):
        if not isinstance(other, ValuationResult):
            raise TypeError(f"Cannot combine ValuationResult with {type(other)}")
        if self.algorithm and self.algorithm != other.algorithm:
            raise ValueError("Cannot combine results from different algorithms")

    def __add__(self, other: ValuationResult) -> ValuationResult:
        """Adds two ValuationResults.

        The values must have been computed with the same algorithm. An exception to this
        is if one argument has empty or all-zero values, in which case the other
        argument is returned.

        !!! danger
            Abusing this will introduce numerical errors.

        Means and standard errors are correctly handled. Statuses are added with
        bit-wise `&`, see [Status][pydvl.valuation.result.Status].
        `data_names` are taken from the left summand, or if unavailable from
        the right one. The `algorithm` string is carried over if both terms
        have the same one or concatenated.

        It is possible to add ValuationResults of different lengths, and with
        different or overlapping indices. The result will have the union of
        indices, and the values.

        !!! Warning
            FIXME: Arbitrary `extra_values` aren't handled.

        """
        self._check_compatible(other)

        if len(self.values) == 0 or (all(self.values == 0.0) and all(self.counts == 0)):
            return other
        if len(other.values) == 0 or (
            all(other.values == 0.0) and all(other.counts == 0)
        ):
            return self

        indices = np.union1d(self._data_indices, other._data_indices).astype(
            self._data_indices.dtype
        )
        this_pos = np.searchsorted(indices, self._data_indices)
        other_pos = np.searchsorted(indices, other._data_indices)

        n: NDArray[np.int_] = np.zeros_like(indices, dtype=int)
        m: NDArray[np.int_] = np.zeros_like(indices, dtype=int)
        xn: NDArray[np.int_] = np.zeros_like(indices, dtype=float)
        xm: NDArray[np.int_] = np.zeros_like(indices, dtype=float)
        vn: NDArray[np.int_] = np.zeros_like(indices, dtype=float)
        vm: NDArray[np.int_] = np.zeros_like(indices, dtype=float)

        n[this_pos] = self._counts
        xn[this_pos] = self._values
        vn[this_pos] = self._variances
        m[other_pos] = other._counts
        xm[other_pos] = other._values
        vm[other_pos] = other._variances

        # np.maximum(1, n + m) covers case n = m = 0.
        n_m_sum = np.maximum(1, n + m)

        # Sample mean of n+m samples from two means of n and m samples
        xnm = (n * xn + m * xm) / n_m_sum

        # Sample variance of n+m samples from two sample variances of n and m samples
        vnm = (n * (vn + xn**2) + m * (vm + xm**2)) / n_m_sum - xnm**2

        if np.any(vnm < 0):
            if np.any(vnm < -1e-6):
                logger.warning(
                    "Numerical error in variance computation. "
                    f"Negative sample variances clipped to 0 in {vnm}"
                )
            vnm[np.where(vnm < 0)] = 0

        # Merging of names:
        # If an index has the same name in both results, it must be the same.
        # If an index has a name in one result but not the other, the name is
        # taken from the result with the name.
        if self._names.dtype != other._names.dtype:
            if np.can_cast(other._names.dtype, self._names.dtype, casting="safe"):
                logger.warning(
                    f"Casting ValuationResult.names from {other._names.dtype} to {self._names.dtype}"
                )
                other._names = other._names.astype(self._names.dtype)
            else:
                raise TypeError(
                    f"Cannot cast ValuationResult.names from "
                    f"{other._names.dtype} to {self._names.dtype}"
                )

        both_pos = np.intersect1d(this_pos, other_pos)

        if len(both_pos) > 0:
            this_names: NDArray = np.empty_like(indices, dtype=np.str_)
            other_names: NDArray = np.empty_like(indices, dtype=np.str_)
            this_names[this_pos] = self._names
            other_names[other_pos] = other._names

            this_shared_names = np.take(this_names, both_pos)
            other_shared_names = np.take(other_names, both_pos)

            if np.any(this_shared_names != other_shared_names):
                raise ValueError("Mismatching names in ValuationResults")

        names = np.empty_like(indices, dtype=self._names.dtype)
        names[this_pos] = self._names
        names[other_pos] = other._names

        return ValuationResult(
            algorithm=self.algorithm or other.algorithm or "",
            status=self.status & other.status,
            indices=indices,
            values=xnm,
            variances=vnm,
            counts=n + m,
            data_names=names,
            # FIXME: What to do with extra_values? This is not commutative:
            # extra_values=self._extra_values.update(other._extra_values),
        )

    def scale(self, factor: float, data_indices: NDArray[IndexT] | None = None):
        """
        Scales the values and variances of the result by a coefficient.

        Args:
            factor: Factor to scale by.
            data_indices: Data indices to scale. If `None`, all values are scaled.
        """
        if data_indices is None:
            positions = None
        else:
            positions = [self._indices[idx] for idx in data_indices]
        self._values[positions] *= factor
        self._variances[positions] *= factor**2

    def get(self, data_idx: IndexT) -> ValueItem:
        """Retrieves a [ValueItem][pydvl.valuation.result.ValueItem] object by data
        index, as opposed to sort index, like the indexing operator.

        Args:
            data_idx: Data index of the value to retrieve.

        Raises:
             IndexError: If the index is not found.
        """
        try:
            pos = self._indices[data_idx]
        except KeyError:
            raise IndexError(f"Index {data_idx} not found in ValuationResult")

        return ValueItem(
            data_idx,
            self._names[pos],
            self._values[pos],
            self._variances[pos],
            self._counts[pos],
        )

    def to_dataframe(
        self, column: str | None = None, use_names: bool = False
    ) -> pd.DataFrame:
        """Returns values as a dataframe.

        Args:
            column: Name for the column holding the data value. Defaults to
                the name of the algorithm used.
            use_names: Whether to use data names instead of indices for the
                DataFrame's index.

        Returns:
            A dataframe with three columns: `name`, `name_variances` and
                `name_counts`, where `name` is the value of argument `column`.
        """
        column = column or self._algorithm
        df = pd.DataFrame(
            self._values[self._positions_to_indices],
            index=(
                self._names[self._positions_to_indices]
                if use_names
                else self._data_indices[self._positions_to_indices]
            ),
            columns=[column],
        )
        df[column + "_variances"] = self.variances[self._positions_to_indices]
        df[column + "_counts"] = self.counts[self._positions_to_indices]
        return df

    @classmethod
    def from_random(
        cls,
        size: int,
        total: float | None = None,
        seed: Seed | None = None,
        **kwargs,
    ) -> ValuationResult:
        """Creates a [ValuationResult][pydvl.valuation.result.ValuationResult] object
        and fills it with an array of random values from a uniform distribution in
        [-1,1]. The values can be made to sum up to a given total number (doing so will
        change their range).

        Args:
            size: Number of values to generate
            total: If set, the values are normalized to sum to this number
                ("efficiency" property of Shapley values).
            seed: Random seed to use
            kwargs: Additional options to pass to the constructor of
                [ValuationResult][pydvl.valuation.result.ValuationResult]. Use to
                override status, names, etc.

        Returns:
            A valuation result with its status set to
            [Status.Converged][pydvl.utils.status.Status] by default.

        Raises:
             ValueError: If `size` is less than 1.

        !!! tip "Changed in version 0.6.0"
            Added parameter `total`. Check for zero size
        """
        if size < 1:
            raise ValueError("Size must be a positive integer")

        rng = np.random.default_rng(seed)
        values = rng.uniform(low=-1, high=1, size=size)
        if total is not None:
            values *= total / np.sum(values)

        options = dict(values=values, status=Status.Converged, algorithm="random")
        options.update(kwargs)
        return cls(**options)  # type: ignore

    @classmethod
    def empty(cls, algorithm: str = "", **kwargs: dict[str, Any]) -> ValuationResult:
        """Creates an empty [ValuationResult][pydvl.valuation.result.ValuationResult]
        object.

        Empty results are characterised by having an empty array of values.

        !!! tip
            When a result is added to an empty one, the empty one is entirely discarded.

        Args:
            algorithm: Name of the algorithm used to compute the values
            kwargs: Additional options to pass to the constructor of
                [ValuationResult][pydvl.valuation.result.ValuationResult]. Use to
                override status, extra_values, etc.
        Returns:
            Object with the results.
        """
        options: dict[str, Any] = dict(
            algorithm=algorithm, status=Status.Pending, values=np.array([])
        )
        return cls(**(options | kwargs))

    @classmethod
    def zeros(
        cls,
        algorithm: str = "",
        indices: IndexSetT | None = None,
        data_names: Sequence[NameT] | NDArray[NameT] | None = None,
        size: int = 0,
        **kwargs: dict[str, Any],
    ) -> ValuationResult:
        """Creates a [ValuationResult][pydvl.valuation.result.ValuationResult] filled
        with zeros.

        !!! info
            When a result is added to a zeroed one, the zeroed one is entirely discarded.

        Args:
            algorithm: Name of the algorithm used to compute the values
            indices: Data indices to use. A copy will be made. If not given,
                the indices will be set to the range `[0, size)`.
            data_names: Data names to use. A copy will be made. If not given,
                the names will be set to the string representation of the indices.
            size: Number of data points whose values are computed. If
                not given, the length of `indices` will be used.
            kwargs: Additional options to pass to the constructor of
                [ValuationResult][pydvl.valuation.result.ValuationResult]. Use to
                override status, extra_values, etc.
        Returns:
            Object with the results.
        """
        indices = cls._create_indices_array(indices, size)
        data_names = cls._create_names_array(data_names, indices)

        options: dict[str, Any] = dict(
            algorithm=algorithm,
            status=Status.Pending,
            indices=indices,
            data_names=data_names,
            values=np.zeros(len(indices)),
            variances=np.zeros(len(indices)),
            counts=np.zeros(len(indices), dtype=np.int_),
        )
        return cls(**(options | kwargs))

    @staticmethod
    def _create_indices_array(
        indices: Sequence[IndexT] | NDArray[IndexT] | None, size: int
    ) -> NDArray[IndexT]:
        if indices is None:
            index_array: NDArray[IndexT] = np.arange(size, dtype=np.int_)
        elif isinstance(indices, np.ndarray):
            index_array = indices.copy()
        else:
            index_array = np.asarray(indices)

        return index_array

    @staticmethod
    def _create_names_array(
        data_names: Sequence[NameT] | NDArray[NameT] | None, indices: NDArray[IndexT]
    ) -> NDArray[NameT]:
        if data_names is None:
            names = np.array(indices, copy=True, dtype=np.str_)
        else:
            names = np.array(data_names, copy=True)

        if len(np.unique(names)) != len(names):
            raise ValueError("Data names must be unique")

        return names


class ResultUpdater(ABC, Generic[ValueUpdateT]):
    """Base class for result updaters.

    A result updater is a strategy to update a valuation result with a value update. It
    is used by the valuation methods to process the
    [ValueUpdates][pydvl.valuation.types.ValueUpdate] emitted by the
    [EvaluationStrategy][pydvl.valuation.samplers.base.EvaluationStrategy] corresponding
    to the sampler.
    """

    def __init__(self, result: ValuationResult):
        self.result = result
        self.n_updates = 0

    @abstractmethod
    def process(self, update: ValueUpdateT) -> ValuationResult: ...


class LogResultUpdater(ResultUpdater[ValueUpdateT]):
    """An object to update valuation results in log-space.

    This updater keeps track of several quantities required to maintain accurate running
    1st and 2nd moments. It also uses the log-sum-exp trick for numerical stability.
    """

    def __init__(self, result: ValuationResult):
        super().__init__(result)
        self._log_sum_positive = np.full_like(result.values, -np.inf)

        pos = result.values > 0
        self._log_sum_positive[pos] = np.log(result.values[pos] * result.counts[pos])
        self._log_sum_negative = np.full_like(result.values, -np.inf)

        neg = result.values < 0
        self._log_sum_negative[neg] = np.log(-result.values[neg] * result.counts[neg])
        self._log_sum2 = np.full_like(result.values, -np.inf)

        nz = result.values != 0
        x2 = (
            result.variances[nz] * np.maximum(1, result.counts[nz] - 1) ** 2
            + result.values[nz] ** 2 * result.counts[nz]
        )
        self._log_sum2[nz] = np.log(x2)

    def process(self, update: ValueUpdate) -> ValuationResult:
        assert update.idx is not None

        try:
            # FIXME: need data index -> fixed index mapping => maybe we need to expose
            #  this in ValuationResult? (this is not result.positions())
            loc: int = self.result._indices[update.idx]
        except KeyError:
            raise IndexError(f"Index {update.idx} not found in ValuationResult")

        self.n_updates += 1
        item = self.result.get(update.idx)

        new_val, new_var, log_sum_pos, log_sum_neg, log_sum2 = log_running_moments(
            self._log_sum_positive[loc].item(),
            self._log_sum_negative[loc].item(),
            self._log_sum2[loc].item(),
            item.count or 0,
            update.log_update,
            new_sign=update.sign,
            unbiased=True,
        )
        self._log_sum_positive[loc] = log_sum_pos
        self._log_sum_negative[loc] = log_sum_neg
        self._log_sum2[loc] = log_sum2

        updated_item = ValueItem(
            idx=item.idx,
            name=item.name,
            value=new_val,
            variance=new_var,
            count=item.count + 1 if item.count is not None else 1,
        )

        self.result.set(item.idx, updated_item)
        return self.result

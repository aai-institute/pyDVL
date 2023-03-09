"""
This module collects types and methods for the inspection of the results of
valuation algorithms.

The most important class is :class:`ValuationResult`, which provides access
to raw values, as well as convenient behaviour as a ``Sequence`` with extended
indexing and updating abilities, and conversion to `pandas DataFrames
<https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.

.. rubric:: Operating on results

Results can be added together with the standard ``+`` operator. Because values
are typically running averages of iterative algorithms, addition behaves like a
weighted average of the two results, with the weights being the number of
updates in each result: adding two results is the same as generating one result
with the mean of the values of the two results as values. The variances are
updated accordingly. See :class:`ValuationResult` for details.

Results can also be sorted by value, variance or number of updates, see
:meth:`ValuationResult.sort`. The arrays of :attr:`ValuationResult.values`,
:attr:`ValuationResult.variances`, :attr:`ValuationResult.counts`,
:attr:`ValuationResult.indices` and :attr:`ValuationResult.names` are sorted in
the same way.

Indexing and slicing of results is supported and :class:`ValueItem` objects are
returned. These objects can be compared with the usual operators, which take
only the :attr:`ValueItem.value` into account.

.. rubric:: Creating result objects

The most commonly used factory method is :meth:`ValuationResult.zeros`, which
creates a result object with all values, variances and counts set to zero.
:meth:`ValuationResult.empty` creates an empty result object, which can be used
as a starting point for adding results together. Empty results are discarded
when added to other results. Finally, :meth:`ValuationResult.from_random`
samples random values uniformly.

"""
from __future__ import annotations

import collections.abc
import logging
from dataclasses import dataclass
from functools import total_ordering
from numbers import Integral
from typing import (
    Any,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
from deprecate import deprecated
from numpy.typing import NDArray

from pydvl.utils.dataset import Dataset
from pydvl.utils.numeric import running_moments
from pydvl.utils.status import Status

try:
    import pandas  # Try to import here for the benefit of mypy
except ImportError:
    pass

__all__ = ["ValuationResult", "ValueItem", "IndexT", "NameT"]

logger = logging.getLogger(__name__)

# TODO: Move to value.types once it's there
IndexT = TypeVar("IndexT", bound=np.int_)
NameT = TypeVar("NameT", bound=Any)


@total_ordering
@dataclass
class ValueItem(Generic[IndexT, NameT]):
    """The result of a value computation for one datum.

    ``ValueItems`` can be compared with the usual operators, forming a total
    order. Comparisons take only the :attr:`value` into account.

    .. todo::
       Maybe have a mode of comparing similar to `np.isclose`, or taking the
       :attr:`variance` into account.
    """

    #: Index of the sample with this value in the original
    #  :class:`~pydvl.utils.dataset.Dataset`
    index: IndexT
    #: Name of the sample if it was provided. Otherwise, `str(index)`
    name: NameT
    #: The value
    value: float
    #: Variance of the value if it was computed with an approximate method
    variance: Optional[float]
    #: Number of updates for this value
    count: Optional[int]

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value

    def __index__(self) -> IndexT:
        return self.index

    @property
    def stderr(self) -> Optional[float]:
        """Standard error of the value."""
        if self.variance is None or self.count is None:
            return None
        return float(np.sqrt(self.variance / self.count).item())


class ValuationResult(
    collections.abc.Sequence, Iterable[ValueItem[IndexT, NameT]], Generic[IndexT, NameT]
):
    """Objects of this class hold the results of valuation algorithms.

    These include indices in the original :class:`Dataset`, any data names (e.g.
    group names in :class:`GroupedDataset`), the values themselves, and variance
    of the computation in the case of Monte Carlo methods. ``ValuationResults``
    can be iterated over like any ``Sequence``: ``iter(valuation_result)``
    returns a generator of :class:`ValueItem` in the order in which the object
    is sorted.

    .. rubric:: Indexing

    Indexing can be position-based, when accessing any of the attributes
    :attr:`values`, :attr:`variances`, :attr:`counts` and :attr:`indices`, as
    well as when iterating over the object, or using the item access operator,
    both getter and setter. The "position" is either the original sequence in
    which the data was passed to the constructor, or the sequence in which the
    object is sorted, see below.

    Alternatively, indexing can be data-based, i.e. using the indices in the
    original dataset. This is the case for the methods :meth:`get` and
    :meth:`update`.

    .. rubric:: Sorting

    Results can be sorted in-place with :meth:`sort`, or alternatively using
    python's standard ``sorted()`` and ``reversed()`` Note that sorting values
    affects how iterators and the object itself as ``Sequence`` behave:
    ``values[0]`` returns a :class:`ValueItem` with the highest or lowest
    ranking point if this object is sorted by descending or ascending value,
    respectively. If unsorted, ``values[0]`` returns the ``ValueItem`` at
    position 0, which has data index ``indices[0]`` in the
    :class:`~pydvl.utils.dataset.Dataset`.

    The same applies to direct indexing of the ``ValuationResult``: the index
    is positional, according to the sorting. It does not refer to the "data
    index". To sort according to data index, use :meth:`sort` with
    ``key="index"``.

    In order to access :class:`ValueItem` objects by their data index, use
    :meth:`get`.

    .. rubric:: Operating on results

    Results can be added to each other with the ``+`` operator. Means and
    variances are correctly updated, using the ``counts`` attribute.

    Results can also be updated with new values using :meth:`update`. Means and
    variances are updated accordingly using the Welford algorithm.

    Empty objects behave in a special way, see :meth:`empty`.

    :param values: An array of values. If omitted, defaults to an empty array
        or to an array of zeros if ``indices`` are given.
    :param indices: An optional array of indices in the original dataset. If
        omitted, defaults to ``np.arange(len(values))``. **Warning:** It is
        common to pass the indices of a :class:`Dataset` here. Attention must be
        paid in a parallel context to copy them to the local process. Just do
        ``indices=np.copy(data.indices)``.
    :param variance: An optional array of variances in the computation of each
        value.
    :param counts: An optional array with the number of updates for each value.
        Defaults to an array of ones.
    :param data_names: Names for the data points. Defaults to index numbers
        if not set.
    :param algorithm: The method used.
    :param status: The end status of the algorithm.
    :param sort: Whether to sort the indices by ascending value. See above how
        this affects usage as an iterable or sequence.
    :param extra_values: Additional values that can be passed as keyword arguments.
        This can contain, for example, the least core value.

    :raise ValueError: If input arrays have mismatching lengths.
    """

    _indices: NDArray[IndexT]
    _values: NDArray[np.float_]
    _counts: NDArray[np.int_]
    _variances: NDArray[np.float_]
    _data: Dataset
    _names: NDArray[NameT]
    _algorithm: str
    _status: Status
    # None for unsorted, True for ascending, False for descending
    _sort_order: Optional[bool]
    _extra_values: dict

    def __init__(
        self,
        *,
        values: NDArray[np.float_],
        variances: Optional[NDArray[np.float_]] = None,
        counts: Optional[NDArray[np.int_]] = None,
        indices: Optional[NDArray[IndexT]] = None,
        data_names: Optional[Sequence[NameT] | NDArray[NameT]] = None,
        algorithm: str = "",
        status: Status = Status.Pending,
        sort: bool = False,
        **extra_values,
    ):
        if variances is not None and len(variances) != len(values):
            raise ValueError("Lengths of values and variances do not match")
        if data_names is not None and len(data_names) != len(values):
            raise ValueError("Lengths of values and data_names do not match")
        if indices is not None and len(indices) != len(values):
            raise ValueError("Lengths of values and indices do not match")

        self._algorithm = algorithm
        self._status = Status(status)  # Just in case we are given a string
        self._values = values
        self._variances = np.zeros_like(values) if variances is None else variances
        self._counts = np.ones_like(values) if counts is None else counts
        self._sort_order = None
        self._extra_values = extra_values or {}

        # Yuk...
        if data_names is None:
            if indices is not None:
                self._names = np.copy(indices)
            else:
                self._names = np.arange(len(self._values), dtype=np.int_)
        elif not isinstance(data_names, np.ndarray):
            self._names = np.array(data_names)
        else:
            self._names = data_names.copy()
        if len(np.unique(self._names)) != len(self._names):
            raise ValueError("Data names must be unique")

        if indices is None:
            indices = np.arange(len(self._values), dtype=np.int_)
        self._indices = indices
        self._positions = {idx: pos for pos, idx in enumerate(indices)}

        self._sort_positions: NDArray[np.int_] = np.arange(
            len(self._values), dtype=np.int_
        )
        if sort:
            self.sort()

    def sort(
        self,
        reverse: bool = False,
        # Need a "Comparable" type here
        key: Literal["value", "variance", "index", "name"] = "value",
    ) -> None:
        """Sorts the indices in place by ``key``.

        Once sorted, iteration over the results, and indexing of all the
        properties :attr:`ValuationResult.values`,
        :attr:`ValuationResult.variances`, :attr:`ValuationResult.counts`,
        :attr:`ValuationResult.indices` and :attr:`ValuationResult.names` will
        follow the same order.

        :param reverse: Whether to sort in descending order by value.
        :param key: The key to sort by. Defaults to :attr:`ValueItem.value`.
        """
        keymap = {
            "index": "_indices",
            "value": "_values",
            "variance": "_variances",
            "name": "_names",
        }
        self._sort_positions = np.argsort(getattr(self, keymap[key]))
        if reverse:
            self._sort_positions = self._sort_positions[::-1]
        self._sort_order = reverse

    @property
    def values(self) -> NDArray[np.float_]:
        """The values, possibly sorted."""
        return self._values[self._sort_positions]

    @property
    def variances(self) -> NDArray[np.float_]:
        """The variances, possibly sorted."""
        return self._variances[self._sort_positions]

    @property
    def stderr(self) -> NDArray[np.float_]:
        """The raw standard errors, possibly sorted."""
        return cast(
            NDArray[np.float_], np.sqrt(self.variances / np.maximum(1, self.counts))
        )

    @property
    def counts(self) -> NDArray[np.int_]:
        """The raw counts, possibly sorted."""
        return self._counts[self._sort_positions]

    @property
    def indices(self) -> NDArray[IndexT]:
        """The indices for the values, possibly sorted.

        If the object is unsorted, then these are the same as declared at
        construction or ``np.arange(len(values))`` if none were passed.
        """
        return self._indices[self._sort_positions]

    @property
    def names(self) -> NDArray[NameT]:
        """The names for the values, possibly sorted.
        If the object is unsorted, then these are the same as declared at
        construction or ``np.arange(len(values))`` if none were passed.
        """
        return self._names[self._sort_positions]

    @property
    def status(self) -> Status:
        return self._status

    @property
    def algorithm(self) -> str:
        return self._algorithm

    def __getattr__(self, attr: str) -> Any:
        """Allows access to extra values as if they were properties of the instance."""
        # This is here to avoid a RecursionError when copying or pickling the object
        if attr == "_extra_values":
            raise AttributeError()
        try:
            return self._extra_values[attr]
        except KeyError as e:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {attr}"
            ) from e

    @overload
    def __getitem__(self, key: int) -> ValueItem:
        ...

    @overload
    def __getitem__(self, key: slice) -> List[ValueItem]:
        ...

    @overload
    def __getitem__(self, key: Iterable[int]) -> List[ValueItem]:
        ...

    def __getitem__(
        self, key: Union[slice, Iterable[int], int]
    ) -> Union[ValueItem, List[ValueItem]]:
        if isinstance(key, slice):
            return [cast(ValueItem, self[i]) for i in range(*key.indices(len(self)))]
        elif isinstance(key, collections.abc.Iterable):
            return [cast(ValueItem, self[i]) for i in key]
        elif isinstance(key, Integral):
            if key < 0:
                key += len(self)
            if key < 0 or int(key) >= len(self):
                raise IndexError(f"Index {key} out of range (0, {len(self)}).")
            idx = self._sort_positions[key]
            return ValueItem(
                self._indices[idx],
                self._names[idx],
                float(self._values[idx]),
                float(self._variances[idx]),
                int(self._counts[idx]),
            )
        else:
            raise TypeError("Indices must be integers, iterable or slices")

    @overload
    def __setitem__(self, key: int, value: ValueItem) -> None:
        ...

    @overload
    def __setitem__(self, key: slice, value: ValueItem) -> None:
        ...

    @overload
    def __setitem__(self, key: Iterable[int], value: ValueItem) -> None:
        ...

    def __setitem__(
        self, key: Union[slice, Iterable[int], int], value: ValueItem
    ) -> None:
        if isinstance(key, slice):
            for i in range(*key.indices(len(self))):
                self[i] = value
        elif isinstance(key, collections.abc.Iterable):
            for i in key:
                self[i] = value
        elif isinstance(key, Integral):
            if key < 0:
                key += len(self)
            if key < 0 or int(key) >= len(self):
                raise IndexError(f"Index {key} out of range (0, {len(self)}).")
            pos = self._sort_positions[key]
            self._indices[pos] = value.index
            self._names[pos] = value.name
            self._values[pos] = value.value
            self._variances[pos] = value.variance
            self._counts[pos] = value.count
        else:
            raise TypeError("Indices must be integers, iterable or slices")

    def __iter__(self) -> Iterator[ValueItem[IndexT, NameT]]:
        """Iterate over the results returning :class:`ValueItem` objects.
        To sort in place before iteration, use :meth:`sort`.
        """
        for pos in self._sort_positions:
            yield ValueItem(
                self._indices[pos],
                self._names[pos],
                self._values[pos],
                self._variances[pos],
                self._counts[pos],
            )

    def __len__(self):
        return len(self._indices)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ValuationResult):
            raise NotImplementedError(
                f"Cannot compare ValuationResult with {type(other)}"
            )
        return bool(
            self._algorithm == other._algorithm
            and self._status == other._status
            and self._sort_order == other._sort_order
            and np.all(self._indices == other._indices)
            and np.all(self._values == other._values)
            and np.all(self._variances == other._variances)
            and np.all(self._names == other._names)
            and np.all(self._counts == other._counts)
        )

    def __repr__(self) -> str:
        repr_string = (
            f"{self.__class__.__name__}("
            f"algorithm='{self._algorithm}',"
            f"status='{self._status.value}',"
            f"values={np.array_str(self.values, precision=4, suppress_small=True)},"
            f"indices={np.array_str(self.indices)},"
            f"names={np.array_str(self.names)},"
            f"counts={np.array_str(self.counts)},"
        )
        for k, v in self._extra_values.items():
            repr_string += f", {k}={v}"
        repr_string += ")"
        return repr_string

    def _check_compatible(self, other: "ValuationResult"):
        if not isinstance(other, ValuationResult):
            raise NotImplementedError(
                f"Cannot combine ValuationResult with {type(other)}"
            )
        if self.algorithm and self.algorithm != other.algorithm:
            raise ValueError("Cannot combine results from different algorithms")

    def __add__(self, other: "ValuationResult") -> "ValuationResult":
        """Adds two ValuationResults.

        The values must have been computed with the same algorithm. An exception
        to this is if one argument has empty values, in which case the other
        argument is returned.

        .. warning::
           Abusing this will introduce numerical errors.

        Means and standard errors are correctly handled. Statuses are added with
        bit-wise ``&``, see :class:`~pydvl.value.result.Status`.
        ``data_names`` are taken from the left summand, or if unavailable from
        the right one. The ``algorithm`` string is carried over if both terms
        have the same one or concatenated.

        It is possible to add ValuationResults of different lengths, and with
        different or overlapping indices. The result will have the union of
        indices, and the values.

        .. warning::
           FIXME: Arbitrary ``extra_values`` aren't handled.

        """
        # empty results
        if len(self.values) == 0:
            return other
        if len(other.values) == 0:
            return self

        self._check_compatible(other)

        indices = np.union1d(self._indices, other._indices).astype(self._indices.dtype)
        this_pos = np.searchsorted(indices, self._indices)
        other_pos = np.searchsorted(indices, other._indices)

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

        # np.maximum(1, n + m) covers case n = m = 0 with
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
                other._names = other._names.astype(self._names.dtype)
                logger.warning(
                    f"Casting ValuationResult.names from {other._names.dtype} to {self._names.dtype}"
                )
            else:
                raise TypeError(
                    f"Cannot cast ValuationResult.names from "
                    f"{other._names.dtype} to {self._names.dtype}"
                )

        both_pos = np.intersect1d(this_pos, other_pos)

        if len(both_pos) > 0:
            this_names: NDArray = np.empty_like(indices, dtype=object)
            other_names: NDArray = np.empty_like(indices, dtype=object)
            this_names[this_pos] = self._names
            other_names[other_pos] = other._names

            this_shared_names = np.take(this_names, both_pos)
            other_shared_names = np.take(other_names, both_pos)

            if np.any(this_shared_names != other_shared_names):
                raise ValueError(f"Mismatching names in ValuationResults")

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

    def update(self, idx: int, new_value: float) -> "ValuationResult":
        """Updates the result in place with a new value, using running mean
        and variance.

        :param idx: Data index of the value to update.
        :param new_value: New value to add to the result.
        :return: A reference to the same, modified result.
        :raises IndexError: If the index is not found.
        """
        try:
            pos = self._positions[idx]
        except KeyError:
            raise IndexError(f"Index {idx} not found in ValuationResult")
        val, var = running_moments(
            self._values[pos], self._variances[pos], self._counts[pos], new_value
        )
        self[pos] = ValueItem(
            index=cast(IndexT, idx),
            name=self._names[pos],
            value=val,
            variance=var,
            count=self._counts[pos] + 1,
        )
        return self

    def scale(self, indices: Optional[NDArray[IndexT]], coefficient: float):
        # Sometimes writeable flag is false on numpy array.
        values = self._values.copy()
        values[self._sort_positions[indices]] *= coefficient
        self._values = values
        variances = self._variances.copy()
        variances[self._sort_positions[indices]] *= coefficient**2
        self._variances = variances

    def get(self, idx: Integral) -> ValueItem:
        """Retrieves a ValueItem by data index, as opposed to sort index, like
        the indexing operator.
        :raises IndexError: If the index is not found.
        """
        try:
            pos = self._positions[idx]
        except KeyError:
            raise IndexError(f"Index {idx} not found in ValuationResult")

        return ValueItem(
            self._indices[pos],
            self._names[pos],
            self._values[pos],
            self._variances[pos],
            self._counts[pos],
        )

    def to_dataframe(
        self, column: Optional[str] = None, use_names: bool = False
    ) -> pandas.DataFrame:
        """Returns values as a dataframe.

        :param column: Name for the column holding the data value. Defaults to
            the name of the algorithm used.
        :param use_names: Whether to use data names instead of indices for the
            DataFrame's index.
        :return: A dataframe with two columns, one for the values, with name
            given as explained in `column`, and another with standard errors for
            approximate algorithms. The latter will be named `column+'_stderr'`.
        :raise ImportError: If pandas is not installed
        """
        if not pandas:
            raise ImportError("Pandas required for DataFrame export")
        column = column or self._algorithm
        df = pandas.DataFrame(
            self._values[self._sort_positions],
            index=self._names[self._sort_positions]
            if use_names
            else self._indices[self._sort_positions],
            columns=[column],
        )
        df[column + "_stderr"] = self.stderr[self._sort_positions]
        return df

    @classmethod
    def from_random(
        cls, size: int, total: Optional[float] = None, **kwargs
    ) -> "ValuationResult":
        """Creates a :class:`ValuationResult` object and fills it with an array
        of random values from a uniform distribution in [-1,1]. The values can
        be made to sum up to a given total number (doing so will change their range).

        :param size: Number of values to generate
        :param total: If set, the values are normalized to sum to this number
            ("efficiency" property of Shapley values).
        :param kwargs: Additional options to pass to the constructor of
            :class:`ValuationResult`. Use to override status, names, etc.
        :return: A valuation result with its status set to
            :attr:`Status.Converged` by default.
        :raises ValueError: If ``size`` is less than 1.

        .. versionchanged:: 0.6.0
            Added parameter ``total``. Check for zero size
        """
        if size < 1:
            raise ValueError("Size must be a positive integer")

        values = np.random.uniform(low=-1, high=1, size=size)
        if total is not None:
            values *= total / np.sum(values)

        options = dict(values=values, status=Status.Converged, algorithm="random")
        options.update(kwargs)
        return cls(**options)  # type: ignore

    @classmethod
    @deprecated(
        target=True,
        deprecated_in="0.6.0",
        remove_in="0.8.0",
        args_mapping=dict(indices=None, data_names=None, n_samples=None),
        template_mgs="`%(source_name)s` is deprecated for generating zero-filled "
        "results, use `ValuationResult.zeros()` instead.",
    )
    def empty(
        cls,
        algorithm: str = "",
        indices: Optional[Sequence[IndexT] | NDArray[IndexT]] = None,
        data_names: Optional[Sequence[NameT] | NDArray[NameT]] = None,
        n_samples: int = 0,
    ) -> "ValuationResult":
        """Creates an empty :class:`ValuationResult` object.

        Empty results are characterised by having an empty array of values. When
        another result is added to an empty one, the empty one is discarded.

        :param algorithm: Name of the algorithm used to compute the values
        :return: An instance of :class:`ValuationResult`
        """
        if indices is not None or data_names is not None or n_samples != 0:
            return cls.zeros(
                algorithm=algorithm,
                indices=indices,
                data_names=data_names,
                n_samples=n_samples,
            )
        return cls(algorithm=algorithm, status=Status.Pending, values=np.array([]))

    @classmethod
    def zeros(
        cls,
        algorithm: str = "",
        indices: Optional[Sequence[IndexT] | NDArray[IndexT]] = None,
        data_names: Optional[Sequence[NameT] | NDArray[NameT]] = None,
        n_samples: int = 0,
    ) -> "ValuationResult":
        """Creates an empty :class:`ValuationResult` object.

        Empty results are characterised by having an empty array of values. When
        another result is added to an empty one, the empty one is ignored.

        :param algorithm: Name of the algorithm used to compute the values
        :param indices: Data indices to use. A copy will be made. If not given,
            the indices will be set to the range ``[0, n_samples)``.
        :param data_names: Data names to use. A copy will be made. If not given,
            the names will be set to the string representation of the indices.
        :param n_samples: Number of data points whose values are computed. If
            not given, the length of ``indices`` will be used.
        :return: An instance of :class:`ValuationResult`
        """
        if indices is None:
            indices = np.arange(n_samples, dtype=np.int_)
        else:
            indices = np.array(indices)
        return cls(
            algorithm=algorithm,
            status=Status.Pending,
            indices=indices,
            data_names=data_names
            if data_names is not None
            else np.empty_like(indices, dtype=object),
            values=np.zeros(len(indices)),
            variances=np.zeros(len(indices)),
            counts=np.zeros(len(indices), dtype=np.int_),
        )

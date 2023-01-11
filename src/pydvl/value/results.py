"""
This module collects types and methods for the inspection of the results of
valuation algorithms.

The most important class is :class:`ValuationResult`, which provides access
to raw values, as well as convenient behaviour as a Sequence with extended
indexing abilities, and conversion to `pandas DataFrames
<https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_.
"""
import collections.abc
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering
from typing import (
    Any,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
    cast,
    overload,
)

import numpy as np
from numpy.typing import NDArray

from pydvl.utils import Dataset

try:
    import pandas  # Try to import here for the benefit of mypy
except ImportError:
    pass

__all__ = ["ValuationResult", "ValuationStatus"]


class ValuationStatus(Enum):
    Pending = "pending"
    Converged = "converged"
    MaxIterations = "maximum number of iterations reached"
    Failed = "failed"


@total_ordering
@dataclass
class ValueItem:
    """The result of a value computation for one datum.

    ValueItems can be compared with the usual operators. These take only the
    :attribute:`value` into account

    .. todo::
       Maybe have a mode of comparing similar to `np.isclose`, or taking the
       :attribute:`stderr` into account.
    """

    #: Index of the sample with this value in the original :class:`Dataset`
    index: np.int_
    #: Name of the sample if it was provided. Otherwise, `str(index)`
    name: str
    #: The value
    value: np.float_
    #: Standard error of the value if it was computed with an approximate method
    stderr: Optional[np.float_]

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value

    def __index__(self):
        return self.index


class ValuationResult(collections.abc.Sequence):
    """Objects of this class hold the results of valuation algorithms.

    These include indices in the original :class:`Dataset`, any data names (e.g.
    group names in :class:`GroupedDataset`), the values themselves, and standard
    errors in the case of Monte Carlo methods. These can iterated over like any
    ``Sequence``: ``iter(valuation_result)`` returns a generator of
    :class:`ValueItem` in the order in which the object is sorted.

    Results can be sorted in-place with :meth:`sort` or using python's standard
    ``sorted()`` and ``reversed()`` Note that sorting values affects how
    iterators and the object itself as ``Sequence`` behave: ``values[0]``
    returns a :class:`ValueItem` with the highest or lowest ranking point if
    this object is sorted by descending or ascending value, respectively. If
    unsorted, ``values[0]`` returns a ``ValueItem`` for index 0.

    :param algorithm: The method used.
    :param status: The end status of the algorithm.
    :param values: An array of values, data indices correspond to positions in
        the array.
    :param stderr: An optional array of standard errors in the computation of
        each value.
    :param data_names: Names for the data points. Defaults to index numbers
        if not set.
    :param sort: Whether to sort the indices by ascending value. See above how
        this affects usage as an iterable or sequence.
    :param extra_values: Additional values that can be passed as keyword arguments.
        This can contain, for example, the least core value.

    :raise ValueError: If data names and values have mismatching lengths.
    """

    _indices: NDArray[np.int_]
    _values: NDArray[np.float_]
    _data: Dataset
    _names: Union[NDArray[np.int_], NDArray[np.str_]]
    _stderr: NDArray[np.float_]
    _algorithm: str  # TODO: BaseValuator
    _status: ValuationStatus  # TODO: Maybe? BaseValuator.Status
    # None for unsorted, True for ascending, False for descending
    _sort_order: Optional[bool]
    _extra_values: dict

    def __init__(
        self,
        algorithm: str,  # BaseValuator,
        status: ValuationStatus,  # Valuation.Status,
        values: NDArray[np.float_],
        stderr: Optional[NDArray[np.float_]] = None,
        data_names: Optional[Sequence[str]] = None,
        sort: bool = True,
        **extra_values,
    ):
        if stderr is not None and len(stderr) != len(values):
            raise ValueError("Lengths of values and stderr do not match")

        self._algorithm = algorithm
        self._status = status
        self._values = values
        self._stderr = np.zeros_like(values) if stderr is None else stderr
        self._sort_order = None
        self._extra_values = extra_values or {}

        if sort:
            self.sort()
        else:
            self._indices = np.arange(0, len(self._values), dtype=np.int_)

        if data_names is None:
            self._names = np.arange(0, len(values), dtype=np.int_)
        else:
            self._names = np.array(data_names)
        if len(self._names) != len(self._values):
            raise ValueError("Data names and data values have different lengths")

    def sort(self, reverse: bool = False) -> None:
        """Sorts the indices in place by ascending value.

        Repeated calls with the same sort order are no-ops. Once sorted,
        iteration over the results will follow the order.

        :param reverse: Whether to sort in descending order by value.
        """

        # Try to save time if we are already sorted in some way
        if self._sort_order is not None:
            if self._sort_order == reverse:  # no change
                return
            self._indices = self._indices[::-1]  # flip order
        else:
            if reverse:
                self._indices = np.argsort(self._values)[::-1]
            else:
                self._indices = np.argsort(self._values)

        self._sort_order = reverse
        return

    @property
    def values(self) -> NDArray[np.float_]:
        """The raw values, unsorted. Position `i` in the array represents index
        `i` of the data."""
        return self._values

    @property
    def indices(self) -> NDArray[np.int_]:
        """The indices for the values, possibly sorted.
        If the object is unsorted, then this is the same as
        `np.arange(len(values))`. Otherwise, the indices sort :meth:`values`
        """
        return self._indices

    @property
    def status(self) -> ValuationStatus:
        return self._status

    @property
    def algorithm(self) -> str:
        return self._algorithm

    def __getattr__(self, attr: str) -> Any:
        """This allows access to extra values as if they were properties of the instance."""
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
        elif isinstance(key, int):
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"Index {key} out of range (0, {len(self)}).")
            idx = self._indices[key]
            return ValueItem(
                idx, self._names[idx], self._values[idx], self._stderr[idx]
            )
        else:
            raise TypeError("Indices must be integers, iterable or slices")

    def __iter__(self) -> Generator[ValueItem, Any, None]:
        """Iterate over the results returning tuples `(index, value)`"""
        for idx in self._indices:
            yield ValueItem(idx, self._names[idx], self._values[idx], self._stderr[idx])

    def __len__(self):
        return len(self._indices)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ValuationResult):
            return NotImplemented
        return bool(
            self._algorithm == other._algorithm
            and self._status == other._status
            and self._sort_order == other._sort_order
            and np.all(self.values == other.values)
            and np.all(self._stderr == other._stderr)
            and np.all(self._names == other._names)
            # and np.all(self.indices == other.indices)  # Redundant
        )

    def __repr__(self) -> str:
        repr_string = (
            f"{self.__class__.__name__}("
            f"algorithm='{self._algorithm}',"
            f"status='{self._status.value}',"
            f"values={np.array_str(self.values, precision=4, suppress_small=True)}"
        )
        for k, v in self._extra_values.items():
            repr_string += f", {k}={v}"
        repr_string += ")"
        return repr_string

    def to_dataframe(
        self, column: Optional[str] = None, use_names: bool = False
    ) -> "pandas.DataFrame":
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
            self._values[self._indices],
            index=self._names[self._indices] if use_names else self._indices,
            columns=[column],
        )

        if self._stderr is None:
            df[column + "_stderr"] = 0
        else:
            df[column + "_stderr"] = self._stderr[self._indices]
        return df

    @classmethod
    def from_random(cls, size: int) -> "ValuationResult":
        """Creates a :class:`ValuationResult` object and fills it
        with an array of random values of the given size uniformly sampled
        from the range [-1, 1].

        :param size: Number of values to generate
        :return: An instance of :class:`ValuationResult`
        """
        values = np.random.uniform(low=-1.0, high=1.0, size=size)
        return cls(
            algorithm="random",
            status=ValuationStatus.Converged,
            values=values,
        )

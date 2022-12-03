import collections
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Union,
)

import numpy as np

from pydvl.utils import Dataset, SortOrder

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["ValuationResult", "ValuationStatus"]


class ValuationStatus(Enum):
    Pending = "pending"
    Converged = "converged"
    MaxIterations = "maximum number of iterations reached"
    Failed = "failed"


class ValueItem(NamedTuple):
    index: np.int_
    name: str
    value: np.float_
    stderr: Optional[np.float_]


class ValuationResult(collections.Sequence):
    """Objects of this class hold the results of valuation algorithms.

    Sorted values affect how iterators and the object itself as sequence behave:
    `values[0]` returns a :class:`ValueItem` with the highest or lowest ranking
    point if this object is sorted by descending or ascending value,
    respectively. If unsorted, `values[0]` returns a `ValueItem` for index 0.

    Similarly `iter(valuation_result)` returns `ValueItem`s in the order in
    which the object is sorted.

    :param algorithm: The method used
    :param status: The end status of the algorithm
    :param values:
    :param stderr:
    :param data_names: Names for the data points. Defaults to index numbers
        if not set.
    :param sort: Whether to sort the values. See above how this affects usage
        as an iterable or sequence.

    :raises ValueError: If data names and values have mismatching lengths.

    .. todo::
       document this

    """

    _indices: "NDArray[np.int_]"
    _values: "NDArray[np.float_]"
    _data: Dataset
    _stderr: Optional["NDArray[np.float_]"]
    _algorithm: str  # TODO: BaseValuator
    _status: ValuationStatus  # TODO: Maybe? BaseValuator.Status
    _sort_order: Optional[SortOrder] = None

    def __init__(
        self,
        algorithm: Callable,  # BaseValuator,
        status: ValuationStatus,  # Valuation.Status,
        values: "NDArray[np.float_]",
        stderr: Optional["NDArray[np.float_]"] = None,
        data_names: Optional[Sequence[str]] = None,
        sort: Optional[SortOrder] = None,
    ):
        if stderr is not None and len(stderr) != len(values):
            raise ValueError("Lengths of values and stderr do not match")

        self._algorithm = getattr(algorithm, "__name__", "value")
        self._status = status
        self._values = values
        self._stderr = stderr

        if data_names is None:
            self._names = np.arange(0, len(values), dtype=np.int_)
        else:
            self._names = np.array(data_names)
        if len(self._names) != len(self._values):
            raise ValueError("Data names and data values have different lengths")
        self.sort(sort)

    def sort(self, sort_order: Optional[SortOrder] = None) -> "ValuationResult":
        """Sorts the values in place.

        Repeated calls with the same `sort_order` are no-ops.

        :param sort_order: None to leave unsorted, otherwise sorts in ascending
            or descending order by value.
        :return: The same object, sorted in place.
        """
        if self._sort_order == sort_order:
            return self

        self._sort_order = sort_order

        if sort_order is None:
            self._indices = np.arange(0, len(self._values), dtype=np.int_)
        else:
            self._indices = np.argsort(self._values)
            if sort_order == SortOrder.Descending:
                self._indices = self._indices[::-1]
        return self

    def is_sorted(self) -> bool:
        return self._sort_order is not None

    @property
    def values(self) -> "NDArray[np.float_]":
        """The raw values, unsorted. Position `i` in the array represents index
        `i` of the data."""
        return self._values

    @property
    def indices(self) -> "NDArray":
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

    def __getitem__(
        self, key: Union[slice, Iterable[int], int]
    ) -> Union[ValueItem, List[ValueItem]]:
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        elif isinstance(key, collections.Iterable):
            return [self[i] for i in key]
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

    def to_dataframe(self, column: Optional[str] = None) -> "DataFrame":
        """Returns values as a dataframe

        :param column: Name for the column holding the data value. Defaults to
            the name of the algorithm used.
        :return: A dataframe with two columns, one for the values, with name
            given as explained in `column`, and another with standard errors for
            approximate algorithms. The latter will be named `column+'_stderr'`.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Pandas required for DataFrame export")

        column = column or self._algorithm
        df = pd.DataFrame(
            self._values[self._indices],
            index=self._names[self._indices],
            columns=[column],
        )

        if self._stderr is None:
            df[column + "_stderr"] = 0
        else:
            df[column + "_stderr"] = self._stderr[self._indices]
        return df

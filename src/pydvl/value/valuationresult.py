from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np

from pydvl.utils import Dataset

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ValuationStatus(Enum):
    Converged = "converged"
    MaxIter = "maximum number of iterations reached"
    Failed = "failed"


class SortOrder(Enum):
    Ascending = "asc"
    Descending = "dsc"


class ValuationResult:
    """Objects of this class hold the results of valuation algorithms.

    .. todo::
       document this

    :param algorithm: The method used
    :param status: The end status of the algorithm
    :param values:
    :param stderr:
    :param data_names: Names for the data points. Defaults to index numbers
        if not set.
    :param sort: Whether to sort the values.

    :raises ValueError: If data names and values have mismatching lengths.
    """

    _indices: "NDArray[np.int_]"
    _values: "NDArray[np.float_]"
    _data: Dataset
    _stderr: Optional["NDArray[np.float_]"]
    _algorithm: str  # TODO: BaseValuator
    _status: ValuationStatus  # TODO: Maybe? BaseValuator.Status

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
        if sort_order is None:
            self._indices = np.arange(0, len(self._values), dtype=np.int_)
        else:
            self._indices = np.argsort(self._values)
            if sort_order == SortOrder.Descending:
                self._indices = self._indices[::-1]
        return self

    @property
    def values(self) -> "NDArray[np.float_]":
        return self._values[self._indices]

    @property
    def names(self) -> "NDArray":
        return self._names[self._indices]

    @property
    def status(self) -> ValuationStatus:
        return self._status

    @property
    def algorithm(self) -> str:
        return self._algorithm

    def to_dataframe(self) -> "DataFrame":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Pandas required for DataFrame export")

        df = pd.DataFrame(
            self._values[self._indices],
            index=self._names[self._indices],
            columns=[self._algorithm],
        )

        if self._stderr is None:
            df[self._algorithm + "_stderr"] = 0
        else:
            df[self._algorithm + "_stderr"] = self._stderr[self._indices]
        return df

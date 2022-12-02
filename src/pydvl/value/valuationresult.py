from enum import Enum
from typing import Callable, Sequence, TYPE_CHECKING, Optional, OrderedDict

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

    """

    _indices: "NDArray[np.int_]"
    _values: "NDArray[np.float_]"
    _data: Dataset
    _stderr: Optional["NDArray[np.float_]"]
    _algorithm: Callable  # TODO: BaseValuator
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
        """

        :param algorithm: The method used
        :param status: The end status of the algorithm
        :param values:
        :param stderr:
        :param data_names: Names for the data points. Defaults to index numbers
            if not set.
        :param sort: Whether to sort the values.

        :raises ValueError: If data names and values have mismatching lengths.
        """
        if stderr and len(stderr) != len(values):
            raise ValueError("Lengths of values and stderr do not match")

        self._algorithm = algorithm
        self._status = status
        self._values = values
        self._stderr = stderr

        if data_names is None:
            self._names = np.arange(0, len(values), dtype=np.int_)
        else:
            self._names = np.array(data_names)

        if sort is None:
            self._indices = np.arange(0, len(values), dtype=np.int_)
        else:
            self._indices = np.argsort(values)
            if sort == SortOrder.Descending:
                self._indices = self._indices[::-1]

        if len(self._names) != len(self._indices):
            raise ValueError("Data names and data values have different lengths")

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
        return str(self._algorithm)

    def to_dataframe(self) -> "DataFrame":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Pandas required for DataFrame export")

        # column = str(self._algorithm)
        df = pd.DataFrame(
            self._values[self._indices],
            index=self._names[self._indices],
            columns=["data_value"],
        )

        if self._stderr is None:
            df["data_value_std"] = 0
        else:
            df["data_value_std"] = pd.Series(self._stderr[self._indices])
        return df

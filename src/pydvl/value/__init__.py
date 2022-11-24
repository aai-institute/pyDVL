r"""
Algorithms for the exact and approximate computation of value and semi-value.

See :ref:`data valuation` for an introduction to the concepts and methods
implemented here.
"""
from typing import TYPE_CHECKING, Optional, OrderedDict, Tuple, Union

import numpy as np

from pydvl.utils import Dataset
from pydvl.value.shapley import ShapleyMode, compute_shapley_values

if TYPE_CHECKING:
    from numpy.typing import NDArray

# from enum import Enum
# class ValueAlgorithms(Enum):
#     pass

# FIXME: temporary
ValueAlgorithms = Union[ShapleyMode, "loo", "Banzhaf"]


class DataValues:
    """Objects of this class hold the results of valuation algorithms.

    .. todo::
       document this

    """

    _indices: "NDArray[np.int_]"
    _values: "NDArray[np.float_]"
    _data: Dataset
    _stderr: Optional["NDArray[np.float_]"]

    def __init__(
        self,
        algorithm: ValueAlgorithms,
        values: "NDArray[np.float_]",
        data: Dataset,
        stderr: Optional["NDArray[np.float_]"] = None,
        sorted: bool = False,
    ):
        if stderr and len(stderr) != len(values):
            raise ValueError("Lengths of values and stderr do not match")

        self._values = values
        self._data = data
        self._stderr = stderr
        self.algorithm = algorithm
        if sorted:
            self._indices = np.arange(0, len(values), dtype=np.int_)
        else:
            self._indices = np.argsort(values)

    def to_dataframe(self) -> "DataFrame":
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Pandas dependency required")

        df = pd.DataFrame(self._values, index=self._indices, columns=["data_value"])

        if self._stderr is None:
            df["data_value_std"] = 0
        else:
            df["data_value_std"] = pd.Series(self._stderr)
        return df

    def to_ordereddict(self) -> Tuple[OrderedDict, OrderedDict]:
        from pydvl.reporting.scores import sort_values

        sorted_values = sort_values(
            {self._data.data_names[i]: v for i, v in enumerate(self._values)}
        )
        if self._stderr:
            sorted_errors = sort_values(
                {self._data.data_names[i]: v for i, v in enumerate(self._stderr)}
            )
        else:
            sorted_errors = sort_values({})

        return sorted_values, sorted_errors

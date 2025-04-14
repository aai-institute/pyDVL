"""
This module contains convenience classes to handle data and groups thereof.

Value computations with supervised models benefit from a unified interface to handle
data. This module provides two classes to handle data and labels, as well as feature
names and other information:

* [Dataset][pydvl.valuation.dataset.Dataset]
* [GroupedDataset][pydvl.valuation.dataset.GroupedDataset].

Objects of both types can be used to construct [scorers][pydvl.valuation.scorers] (for
the valuation set) and to `fit` (most) valuation methods.

The underlying data arrays can always be accessed (read-only) via
[Dataset.data()][pydvl.valuation.dataset.Dataset.data], which returns the tuple `(x,
y)`.

!!! tip "Logical vs data indices"
    [Dataset][pydvl.valuation.dataset.Dataset] and
    [GroupedDataset][pydvl.valuation.dataset.GroupedDataset] use two different types of
    indices:
    * **Logical indices**: These are the indices used to access the elements in the
      `(Grouped)Dataset` objects when slicing or indexing them.
    * **Data indices**: These are the indices used to access the data points in the
        underlying data arrays. They are the indices used in the
        [RawData][pydvl.valuation.dataset.RawData] object returned by
        [Dataset.data()][pydvl.valuation.dataset.Dataset.data].

## Slicing

Slicing a [Dataset][] object, e.g. `dataset[0]`, will return a new `Dataset` with the
data corresponding to that slice. Note however that the contents of the new object, i.e.
`dataset[0].data().x`, may not be the same as `dataset.data().x[0]`, which is the first
point in the original data array. This is in particular true for
[GroupedDatasets][pydvl.valuation.dataset.GroupedDataset] where one "logical" index may
correspond to multiple data points.

Slicing with `None`, i.e. `dataset[None]`, will return a copy of the whole dataset.

## Grouped datasets and logical indices

As mentioned above, it is also possible to group data points together with
[GroupedDataset][pydvl.valuation.dataset.GroupedDataset]. In order to handle groups
correctly, Datasets map "logical" indices to "data" indices and vice versa. The latter
correspond to indices in the data arrays themselves, while the former may map to groups
of data points.

A call to [GroupedDataset.data(indices)][pydvl.valuation.dataset.GroupedDataset.data]
will return the data and labels of all samples for the given groups. But
`grouped_data[0]` will return the data and labels of the first group, not the first data
point and will therefore be in general different from `grouped_data.data([0])`.

Grouping data can be useful to reduce computation time, e.g. for Shapley-based methods,
or to look at the importance of certain feature sets for the model.

!!! tip
    It is important to keep in mind the distinction between logical and data indices for
    valuation methods that require computation on individual data points, like
    [KNNShapleyValuation][pydvl.valuation.methods.knn_shapley.KNNShapleyValuation] or
    [DataOOBValuation][pydvl.valuation.methods.data_oob.DataOOBValuation]. In these
    cases, the logical indices are used to compute the Shapley values, while the data
    indices are used internally by the method.

!!! tip "New in version 0.11.0"
    Added support for PyTorch tensors. The Dataset and GroupedDataset classes now
    work with both
    NumPy arrays and PyTorch tensors, preserving the input type throughout operations.
"""

from __future__ import annotations

import atexit
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Generic, Sequence, overload

import numpy as np
from deprecate import deprecated
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

__all__ = ["Dataset", "GroupedDataset", "RawData"]

from pydvl.utils import is_numpy, is_tensor
from pydvl.utils.types import ArrayT, atleast1d, check_X_y

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawData(Generic[ArrayT]):
    """A view on a dataset's raw data. This is not a copy."""

    x: ArrayT
    y: ArrayT

    def __post_init__(self):
        # Check that x and y are valid arrays (either numpy arrays or torch tensors)
        if not (
            (is_numpy(self.x) and is_numpy(self.y))
            or (is_tensor(self.x) and is_tensor(self.y))
        ):
            raise TypeError(
                "x and y must both be valid arrays of the same type (numpy arrays "
                "or torch tensors)"
            )

        # Check that x and y have matching types
        if is_numpy(self.x) != is_numpy(self.y):
            raise TypeError("x and y must be arrays of the same type")

        # Check that x and y have matching first dimensions
        try:
            if self.x.shape[0] != self.y.shape[0]:
                raise ValueError("x and y must have the same length")
        except (TypeError, AttributeError) as e:
            raise TypeError("x and y must be valid arrays with shape attribute") from e

    # Make the unpacking operator work
    def __iter__(self):  # No way to type the return Iterator properly
        return iter((self.x, self.y))

    def __getitem__(self, item: int | slice | Sequence[int]) -> RawData:
        return RawData(atleast1d(self.x[item]), atleast1d(self.y[item]))

    def __len__(self) -> int:
        return len(self.x)


class Dataset(Generic[ArrayT]):
    """A convenience class to handle datasets.

    It holds a dataset, together with info on feature names, target names, and
    data names. It is used to pass data around to valuation methods.

    The underlying data arrays can be accessed via
    [Dataset.data()][pydvl.valuation.dataset.Dataset.data], which returns the tuple
    `(X, y)` as a read-only [RawData][pydvl.valuation.dataset.RawData] object. The data
    can be accessed by indexing the object directly, e.g. `dataset[0]` will return the
    data point corresponding to index 0 in `dataset`. For this base class, this is the
    same as `dataset.data([0])`, which is the first point in the data array, but derived
    classes can behave differently.

    Args:
        x: training data (NumPy array or PyTorch tensor)
        y: labels for training data (same type as x)
        feature_names: names of the features of x data
        target_names: names of the features of y data
        data_names: names assigned to data points.
            For example, if the dataset is a time series, each entry can be a
            timestamp which can be referenced directly instead of using a row
            number.
        description: A textual description of the dataset.
        multi_output: set to `False` if labels are scalars, or to
            `True` if they are vectors of dimension > 1.
        mmap: Whether to use memory-mapped files (for NumPy arrays only)

    !!! tip "Changed in version 0.10.0"
        No longer holds split data, but only x, y.
    !!! tip "Changed in version 0.10.0"
        Slicing now return a new `Dataset` object, not raw data.
    !!! tip "New in version 0.11.0"
        Added support for PyTorch tensors.
    """

    _indices: NDArray[np.int_]
    _data_names: NDArray[np.str_]
    feature_names: list[str]
    target_names: list[str]
    _x: ArrayT
    _y: ArrayT

    def __init__(
        self,
        x: ArrayT,
        y: ArrayT,
        feature_names: Sequence[str] | NDArray[np.str_] | None = None,
        target_names: Sequence[str] | NDArray[np.str_] | None = None,
        data_names: Sequence[str] | NDArray[np.str_] | None = None,
        description: str | None = None,
        multi_output: bool = False,
        mmap: bool = False,
    ):
        if mmap:
            if not is_numpy(x) or not is_numpy(y):
                raise TypeError("x and y must be numpy arrays in order to use mmap")

            _x, _y = check_X_y(x, y, multi_output=multi_output, estimator="Dataset")
            tmpdir = mkdtemp()
            logger.debug(f"Using temporary directory {tmpdir} for mmap files")
            path_x = Path(tmpdir) / "dataset-x.mmap"
            path_y = Path(tmpdir) / "dataset-y.mmap"
            fp_x = np.memmap(path_x, dtype=_x.dtype, mode="w+", shape=_x.shape)
            fp_y = np.memmap(path_y, dtype=_y.dtype, mode="w+", shape=_y.shape)
            fp_x[:] = _x[:]
            fp_y[:] = _y[:]
            self._x_dtype = _x.dtype
            self._y_dtype = _y.dtype
            self._x_shape = _x.shape
            self._y_shape = _y.shape
            self._x = np.memmap(path_x, dtype=_x.dtype, mode="readonly", shape=_x.shape)  # type: ignore
            self._y = np.memmap(path_y, dtype=_y.dtype, mode="readonly", shape=_y.shape)  # type: ignore
            del fp_x, fp_y  # note this does not close the mmaps
            del _x, _y

            def cleanup_mmap_files():
                try:
                    path_x.unlink(missing_ok=True)
                    path_y.unlink(missing_ok=True)
                    logger.debug("Temporary mmap files deleted successfully.")
                except Exception as e:
                    logger.error(f"Error while deleting mmap files: {e}")

            atexit.register(cleanup_mmap_files)
        else:
            self._x, self._y = check_X_y(
                x, y, multi_output=multi_output, estimator="Dataset"
            )

        def make_names(s: str, a: ArrayT) -> list[str]:
            n = a.shape[1] if len(a.shape) > 1 else 1
            return [f"{s}{i:0{1 + int(math.log10(n))}d}" for i in range(1, n + 1)]

        self.feature_names = (
            list(feature_names) if feature_names is not None else make_names("x", x)
        )
        self.target_names = (
            list(target_names) if target_names is not None else make_names("y", y)
        )

        if len(self._x.shape) > 1:
            if len(self.feature_names) != self._x.shape[1]:
                raise ValueError("Mismatching number of features and names")
        if len(self._y.shape) > 1:
            if len(self.target_names) != self._y.shape[1]:
                raise ValueError("Mismatching number of targets and names")

        self.description = description or "No description"

        # Always use numpy arrays for indices
        self._indices = np.arange(len(self._x), dtype=np.int_)

        # Generate data names if not provided
        if data_names is not None:
            self._data_names = np.array(data_names, dtype=np.str_)
        else:
            self._data_names = self._indices.astype(np.str_)

    def __getstate__(self):
        """Prepare the object state for pickling."""
        state = self.__dict__.copy()
        # Replace memmapped arrays with their file paths
        if isinstance(self._x, np.memmap):
            state["_x"] = Path(self._x.filename)
        if isinstance(self._y, np.memmap):
            state["_y"] = Path(self._y.filename)
        return state

    def __setstate__(self, state):
        """Restore the object state from pickling."""
        self.__dict__.update(state)

        if isinstance(self._x, Path):
            self._x = np.memmap(
                self._x, dtype=self._x_dtype, mode="readonly", shape=self._x_shape
            )
        if isinstance(self._y, (str, Path)):
            self._y = np.memmap(
                self._y, dtype=self._y_dtype, mode="readonly", shape=self._y_shape
            )

    def __getitem__(
        self, idx: int | slice | Sequence[int] | NDArray[np.int_]
    ) -> Dataset:
        if idx is None:
            idx = slice(None)
        elif isinstance(idx, int):
            idx = [idx]
        return Dataset(
            x=self._x[idx],
            y=self._y[idx],
            feature_names=self.feature_names,
            target_names=self.target_names,
            data_names=self._data_names[idx],
            description="(SLICED): " + self.description,
        )

    def feature(self, name: str) -> tuple[slice, int]:
        """Returns a slice for the feature with the given name."""
        try:
            return np.index_exp[:, self.feature_names.index(name)]  # type: ignore
        except ValueError:
            raise ValueError(f"Feature {name} is not in {self.feature_names}")

    def data(
        self, indices: int | slice | Sequence[int] | NDArray[np.int_] | None = None
    ) -> RawData:
        """Given a set of indices, returns the training data that refer to those
        indices, as a read-only tuple-like structure.

        This is used mainly by subclasses of
        [UtilityBase][pydvl.valuation.utility.base.UtilityBase] to retrieve subsets of
        the data from indices.

        Args:
            indices: Optional indices that will be used to select points from
                the training data. If `None`, the entire training data will be
                returned.

        Returns:
            If `indices` is not `None`, the selected x and y arrays from the
                training data. Otherwise, the entire dataset.
        """
        if indices is None:
            return RawData(self._x, self._y)
        return RawData(self._x[indices], self._y[indices])

    def data_indices(self, indices: Sequence[int] | None = None) -> NDArray[np.int_]:
        """Returns a subset of indices.

        This is equivalent to using `Dataset.indices[logical_indices]` but allows
        subclasses to define special behaviour, e.g. when indices in `Dataset` do not
        match the indices in the data.

        For `Dataset`, this is a simple pass-through.

        Args:
            indices: A set of indices held by this object

        Returns:
            The indices of the data points in the data array.
        """
        if indices is None:
            return self._indices
        return self._indices[indices]

    def logical_indices(self, indices: Sequence[int] | None = None) -> NDArray[np.int_]:
        """Returns the indices in this `Dataset` for the given indices in the data
        array.

        This is equivalent to using `Dataset.indices[data_indices]` but allows
        subclasses to define special behaviour, e.g. when indices in `Dataset` do not
        match the indices in the data.

        Args:
            indices: A set of indices in the data array.

        Returns:
            The abstract indices for the given data indices.
        """
        if indices is None:
            return self._indices
        return self._indices[indices]

    def target(self, name: str) -> tuple[slice, int]:
        try:
            return np.index_exp[:, self.target_names.index(name)]  # type: ignore
        except ValueError:
            raise ValueError(f"Target {name} is not in {self.target_names}")

    @property
    def indices(self) -> NDArray[np.int_]:
        """Index of positions in data.x_train.

        Contiguous integers from 0 to len(Dataset).
        """
        return self._indices

    @property
    def names(self) -> NDArray[np.str_]:
        """Names of each individual datapoint.

        Used for reporting Shapley values.
        """
        return self._data_names

    @property
    def n_features(self) -> int:
        """Returns the number of dimensions of a sample."""
        return int(self._x.shape[1]) if len(self._x.shape) > 1 else 1

    @property
    @deprecated(
        target=None,  # cannot set to Dataset.n_features
        deprecated_in="0.10.0",
        remove_in="0.11.0",
    )
    def dim(self):
        return self.n_features

    def __str__(self):
        return self.description

    def __len__(self):
        return len(self._x)

    @classmethod
    def from_sklearn(
        cls,
        data: Bunch,
        train_size: int | float = 0.8,
        random_state: int | None = None,
        stratify_by_target: bool = False,
        **kwargs,
    ) -> tuple[Dataset, Dataset]:
        """Constructs two [Dataset][pydvl.valuation.dataset.Dataset] objects from a
        [sklearn.utils.Bunch][], as returned by the `load_*`
        functions in [scikit-learn toy datasets](
        https://scikit-learn.org/stable/datasets/toy_dataset.html).

        ??? Example
            ```pycon
            >>> from pydvl.valuation.dataset import Dataset
            >>> from sklearn.datasets import load_boston  # noqa
            >>> train, test = Dataset.from_sklearn(load_boston())
            ```

        Args:
            data: scikit-learn Bunch object. The following attributes are supported:

                - `data`: covariates.
                - `target`: target variables (labels).
                - `feature_names` (**optional**): the feature names.
                - `target_names` (**optional**): the target names.
                - `DESCR` (**optional**): a description.
            train_size: size of the training dataset. Used in `train_test_split`
                float values represent the fraction of the dataset to include in the
                training split and should be in (0,1). An integer value sets the
                absolute number of training samples.
        the value is automatically set to the complement of the test size.
            random_state: seed for train / test split
            stratify_by_target: If `True`, data is split in a stratified
                fashion, using the target variable as labels. Read more in
                [scikit-learn's user guide](
                https://scikit-learn.org/stable/modules/cross_validation.html
                #stratification).
            kwargs: Additional keyword arguments to pass to the
                [Dataset][pydvl.valuation.dataset.Dataset] constructor. Use this to
                pass e.g. `is_multi_output`.

        Returns:
            Object with the sklearn dataset

        !!! tip "Changed in version 0.6.0"
            Added kwargs to pass to the [Dataset][pydvl.valuation.dataset.Dataset]
            constructor.
        !!! tip "Changed in version 0.10.0"
            Returns a tuple of two [Dataset][pydvl.valuation.dataset.Dataset] objects.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            train_size=train_size,
            random_state=random_state,
            stratify=data.target if stratify_by_target else None,
        )
        return (
            cls(
                x_train,
                y_train,
                feature_names=data.get("feature_names"),
                target_names=data.get("target_names"),
                description=data.get("DESCR"),
                **kwargs,
            ),
            cls(
                x_test,
                y_test,
                feature_names=data.get("feature_names"),
                target_names=data.get("target_names"),
                description=data.get("DESCR"),
                **kwargs,
            ),
        )

    @classmethod
    def from_arrays(
        cls,
        X: ArrayT,
        y: ArrayT,
        train_size: float = 0.8,
        random_state: int | None = None,
        stratify_by_target: bool = False,
        **kwargs: Any,
    ) -> tuple[Dataset, Dataset]:
        """Constructs a [Dataset][pydvl.valuation.dataset.Dataset] object from X and
        y arrays as returned by the `make_*` functions in [sklearn generated datasets](
        https://scikit-learn.org/stable/datasets/sample_generators.html).

        ??? Example
            ```pycon
            >>> from pydvl.valuation.dataset import Dataset
            >>> from sklearn.datasets import make_regression
            >>> X, y = make_regression()
            >>> dataset = Dataset.from_arrays(X, y)
            ```

        Args:
            X: array of shape (n_samples, n_features) - either numpy array or PyTorch tensor
            y: array of shape (n_samples,) - must be same type as X
            train_size: size of the training dataset. Used in `train_test_split`
            random_state: seed for train / test split
            stratify_by_target: If `True`, data is split in a stratified fashion,
                using the y variable as labels. Read more in [sklearn's user
                guide](https://scikit-learn.org/stable/modules/cross_validation.html
                #stratification).
            kwargs: Additional keyword arguments to pass to the
                [Dataset][pydvl.valuation.dataset.Dataset] constructor. Use this to pass
                e.g. `feature_names` or `target_names`.

        Returns:
            Object with the passed X and y arrays split across training and test sets.

        !!! tip "New in version 0.4.0"

        !!! tip "Changed in version 0.6.0"
            Added kwargs to pass to the [Dataset][pydvl.valuation.dataset.Dataset]
            constructor.

        !!! tip "Changed in version 0.10.0"
            Returns a tuple of two [Dataset][pydvl.valuation.dataset.Dataset] objects.
            
        !!! tip "New in version 0.11.0"
            Added support for PyTorch tensors.
        """
        if stratify_by_target:
            # Use our stratified_split_indices function for proper tensor handling
            train_indices, test_indices = stratified_split_indices(
                y, train_size=train_size, random_state=random_state
            )
            
            # Split data using the indices
            x_train, y_train = X[train_indices], y[train_indices]
            x_test, y_test = X[test_indices], y[test_indices]
        else:
            # For non-stratified splits
            x_train, x_test, y_train, y_test = train_test_split(
                X,
                y,
                train_size=train_size,
                random_state=random_state,
            )
            
        return cls(x_train, y_train, **kwargs), cls(x_test, y_test, **kwargs)


class GroupedDataset(Dataset):
    _group_names: NDArray[np.str_]

    def __init__(
        self,
        x: ArrayT,
        y: ArrayT,
        data_groups: Sequence[int] | NDArray[np.int_],
        feature_names: Sequence[str] | NDArray[np.str_] | None = None,
        target_names: Sequence[str] | NDArray[np.str_] | None = None,
        data_names: Sequence[str] | NDArray[np.str_] | None = None,
        group_names: Sequence[str] | NDArray[np.str_] | None = None,
        description: str | None = None,
        **kwargs: Any,
    ):
        """Class for grouping datasets.

        Used for calculating values of subsets of the data considered as logical units.
        For instance, one can group by value of a categorical feature, by bin into which
        a continuous feature falls, or by label.

        Args:
            x: training data
            y: labels of training data
            data_groups: Sequence of the same length as `x_train` containing
                a group id for each training data point. Data points with the same
                id will then be grouped by this object and considered as one for
                effects of valuation. Group ids are assumed to be zero-based consecutive
                integers
            feature_names: names of the covariates' features.
            target_names: names of the labels or targets y
            data_names: names of the data points. For example, if the dataset is a
                time series, each entry can be a timestamp.
            group_names: names of the groups. If not provided, the numerical group ids
                from `data_groups` will be used.
            description: A textual description of the dataset
            kwargs: Additional keyword arguments to pass to the
                [Dataset][pydvl.valuation.dataset.Dataset] constructor.

        !!! tip "Changed in version 0.6.0"
            Added `group_names` and forwarding of `kwargs`

        !!! tip "Changed in version 0.10.0"
            No longer holds split data, but only x, y and group information. Added
                methods to retrieve indices for groups and vice versa.
        """
        super().__init__(
            x=x,
            y=y,
            feature_names=feature_names,
            target_names=target_names,
            data_names=data_names,
            description=description,
            **kwargs,
        )

        if len(data_groups) != len(x):
            raise ValueError(
                f"data_groups and x must have the same length."
                f"Instead got {len(data_groups)=} and {len(x)=}"
            )

        # data index -> abstract index (group id)
        try:
            self.data_to_group: NDArray[np.int_] = np.array(data_groups, dtype=int)
        except ValueError as e:
            raise ValueError(
                "data_groups must be a mapping from integer data indices to "
                "integer group ids"
            ) from e
        # abstract index (group id) -> data index
        self.group_to_data: OrderedDict[int, list[int]] = OrderedDict(
            {k: [] for k in set(data_groups)}
        )
        for data_idx, group_idx in enumerate(self.data_to_group):
            self.group_to_data[group_idx].append(data_idx)  # type: ignore
        self._indices = np.array(list(self.group_to_data.keys()), dtype=np.int_)
        self._group_names = (
            np.array(group_names, dtype=np.str_)
            if group_names is not None
            else np.array(list(self.group_to_data.keys()), dtype=np.str_)
        )
        if len(self._group_names) != len(self.group_to_data):
            raise ValueError(
                f"The number of group names ({len(self._group_names)}) "
                f"does not match the number of groups ({len(self.group_to_data)})"
            )

    def __len__(self):
        return len(self._indices)

    def __getitem__(
        self, idx: int | slice | Sequence[int] | NDArray[np.int_]
    ) -> GroupedDataset:
        if isinstance(idx, int):
            idx = [idx]
        indices = self.data_indices(idx)
        return GroupedDataset(
            x=self._x[indices],
            y=self._y[indices],
            data_groups=self.data_to_group[indices],
            feature_names=self.feature_names,
            target_names=self.target_names,
            data_names=self._data_names[indices],
            group_names=self._group_names[idx],
            description="(SLICED): " + self.description,
        )

    @property
    def indices(self):
        """Indices of the groups."""
        return self._indices

    @property
    def names(self) -> NDArray[np.str_]:
        """Names of the groups."""
        # FIXME? this shadows _data_names (but it can still be accessed...)
        return self._group_names

    def data(
        self, indices: int | slice | Sequence[int] | NDArray[np.int_] | None = None
    ) -> RawData:
        """Returns the data and labels of all samples in the given groups.

        Args:
            indices: group indices whose elements to return. If `None`,
                all data from all groups are returned.

        Returns:
            Tuple of training data `x` and labels `y`.
        """
        return super().data(self.data_indices(indices))

    def data_indices(
        self, indices: int | slice | Sequence[int] | NDArray[np.int_] | None = None
    ) -> NDArray[np.int_]:
        """Returns the indices of the samples in the given groups.

        Args:
            indices: group indices whose elements to return. If `None`,
                all indices from all groups are returned.

        Returns:
            Indices of the samples in the given groups.
        """
        if indices is None:
            indices = self._indices
        if isinstance(indices, slice):
            indices = range(*indices.indices(len(self.group_to_data)))
        return np.concatenate([self.group_to_data[i] for i in indices], dtype=np.int_)  # type: ignore

    def logical_indices(self, indices: Sequence[int] | None = None) -> NDArray[np.int_]:
        """Returns the group indices for the given data indices.

        Args:
            indices: indices of the data points in the data array. If `None`,
                the group indices for all data points are returned.

        Returns:
            Group indices for the given data indices.
        """
        if indices is None:
            return self.data_to_group
        return self.data_to_group[indices]

    @overload
    @classmethod
    def from_sklearn(
        cls,
        data: Bunch,
        train_size: float = 0.8,
        random_state: int | None = None,
        stratify_by_target: bool = False,
        **kwargs,
    ) -> tuple[GroupedDataset, GroupedDataset]: ...

    @overload
    @classmethod
    def from_sklearn(
        cls,
        data: Bunch,
        train_size: float = 0.8,
        random_state: int | None = None,
        stratify_by_target: bool = False,
        data_groups: Sequence[int] | None = None,
        **kwargs,
    ) -> tuple[GroupedDataset, GroupedDataset]: ...

    @classmethod
    def from_sklearn(
        cls,
        data: Bunch,
        train_size: int | float = 0.8,
        random_state: int | None = None,
        stratify_by_target: bool = False,
        data_groups: Sequence[int] | None = None,
        **kwargs: dict[str, Any],
    ) -> tuple[GroupedDataset, GroupedDataset]:
        """Constructs a [GroupedDataset][pydvl.valuation.dataset.GroupedDataset]
        object, and an
        ungrouped [Dataset][pydvl.valuation.dataset.Dataset] object from a
        [sklearn.utils.Bunch][sklearn.utils.Bunch] as returned by the `load_*`
        functions in
        [scikit-learn toy datasets](
        https://scikit-learn.org/stable/datasets/toy_dataset.html) and groups
        it.

        ??? Example
            ```pycon
            >>> from sklearn.datasets import load_iris
            >>> from pydvl.valuation.dataset import GroupedDataset
            >>> iris = load_iris()
            >>> data_groups = iris.test_data[:, 0] // 0.5
            >>> train, test = GroupedDataset.from_sklearn(iris, data_groups=data_groups)
            ```

        Args:
            data: scikit-learn Bunch object. The following attributes are supported:
                - `data`: covariates.
                - `target`: target variables (labels).
                - `feature_names` (**optional**): the feature names.
                - `target_names` (**optional**): the target names.
                - `DESCR` (**optional**): a description.
            train_size: size of the training dataset. Used in `train_test_split`
                float values represent the fraction of the dataset to include in the
                training split and should be in (0,1). An integer value sets the
                absolute number of training samples.
            random_state: seed for train / test split.
            stratify_by_target: If `True`, data is split in a stratified
                fashion, using the target variable as labels. Read more in
                [sklearn's user guide](
                https://scikit-learn.org/stable/modules/cross_validation.html
                #stratification).
            data_groups: an array holding the group index or name for each
                data point. The length of this array must be equal to the number of
                data points in the dataset.
            kwargs: Additional keyword arguments to pass to the
                [Dataset][pydvl.valuation.dataset.Dataset] constructor.

        Returns:
            Datasets with the selected sklearn data

        !!! tip "Changed in version 0.10.0"
            Returns a tuple of two [GroupedDataset][
            pydvl.valuation.dataset.GroupedDataset]
                objects.
        """

        return cls.from_arrays(
            X=data.data,
            y=data.target,
            train_size=train_size,
            random_state=random_state,
            stratify_by_target=stratify_by_target,
            data_groups=data_groups,
            **kwargs,
        )

    @overload
    @classmethod
    def from_arrays(
        cls,
        X: ArrayT,
        y: ArrayT,
        train_size: float = 0.8,
        random_state: int | None = None,
        stratify_by_target: bool = False,
        **kwargs,
    ) -> tuple[GroupedDataset, GroupedDataset]: ...

    @overload
    @classmethod
    def from_arrays(
        cls,
        X: ArrayT,
        y: ArrayT,
        train_size: float = 0.8,
        random_state: int | None = None,
        stratify_by_target: bool = False,
        data_groups: Sequence[int] | None = None,
        **kwargs,
    ) -> tuple[GroupedDataset, GroupedDataset]: ...

    @classmethod
    def from_arrays(
        cls,
        X: ArrayT,
        y: ArrayT,
        train_size: float = 0.8,
        random_state: int | None = None,
        stratify_by_target: bool = False,
        data_groups: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> tuple[GroupedDataset, GroupedDataset]:
        """Constructs a [GroupedDataset][pydvl.valuation.dataset.GroupedDataset] object,
        and an ungrouped [Dataset][pydvl.valuation.dataset.Dataset] object from X and y
        numpy arrays as returned by the `make_*` functions in
        [scikit-learn generated datasets](
        https://scikit-learn.org/stable/datasets/sample_generators.html).

        ??? Example
            ```pycon
            >>> from sklearn.datasets import make_classification
            >>> from pydvl.valuation.dataset import GroupedDataset
            >>> X, y = make_classification(
            ...     n_samples=100,
            ...     n_features=4,
            ...     n_informative=2,
            ...     n_redundant=0,
            ...     random_state=0,
            ...     shuffle=False
            ... )
            >>> data_groups = X[:, 0] // 0.5
            >>> train, test = GroupedDataset.from_arrays(X, y, data_groups=data_groups)
            ```

        Args:
            X: array of shape (n_samples, n_features)
            y: array of shape (n_samples,)
            train_size: size of the training dataset. Used in `train_test_split`.
            random_state: seed for train / test split.
            stratify_by_target: If `True`, data is split in a stratified
                fashion, using the y variable as labels. Read more in
                [sklearn's user guide](
                https://scikit-learn.org/stable/modules/cross_validation.html
                #stratification).
            data_groups: an array holding the group index or name for each data
                point. The length of this array must be equal to the number of
                data points in the dataset.
            kwargs: Additional keyword arguments that will be passed to the
                [GroupedDataset][pydvl.valuation.dataset.GroupedDataset] constructor.

        Returns:
            Dataset with the passed X and y arrays split across training and
                test sets.

        !!! tip "New in version 0.4.0"

        !!! tip "Changed in version 0.6.0"
            Added kwargs to pass to the
                [GroupedDataset][pydvl.valuation.dataset.GroupedDataset] constructor.

        !!! tip "Changed in version 0.10.0"
            Returns a tuple of two
                [GroupedDataset][pydvl.valuation.dataset.GroupedDataset] objects.
        """

        if data_groups is None:
            raise ValueError(
                "data_groups must be provided when constructing a GroupedDataset"
            )
        x_train, x_test, y_train, y_test, groups_train, groups_test = train_test_split(
            X,
            y,
            data_groups,
            train_size=train_size,
            random_state=random_state,
            stratify=y if stratify_by_target else None,
        )
        training_set = cls(x=x_train, y=y_train, data_groups=groups_train, **kwargs)
        test_set = cls(x=x_test, y=y_test, data_groups=groups_test, **kwargs)
        return training_set, test_set

    @classmethod
    def from_dataset(
        cls,
        data: Dataset,
        data_groups: Sequence[int] | NDArray[np.int_],
        group_names: Sequence[str] | NDArray[np.str_] | None = None,
        **kwargs: Any,
    ) -> GroupedDataset:
        """Creates a [GroupedDataset][pydvl.valuation.dataset.GroupedDataset] object
        from a
        [Dataset][pydvl.valuation.dataset.Dataset] object and a mapping of data groups.

        ??? Example
            ```pycon
            >>> import numpy as np
            >>> from pydvl.valuation.dataset import Dataset, GroupedDataset
            >>> train, test = Dataset.from_arrays(
            ...     X=np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]]),
            ...     y=np.asarray([0, 1, 0, 1]),
            ... )
            >>> grouped_train = GroupedDataset.from_dataset(train, data_groups=[0, 0,
            1, 1])
            ```

        Args:
            data: The original data.
            data_groups: An array holding the group index or name for each data
                point. The length of this array must be equal to the number of
                data points in the dataset.
            group_names: Names of the groups. If not provided, the numerical group ids
                from `data_groups` will be used.
            kwargs: Additional arguments to be passed to the
                [GroupedDataset][pydvl.valuation.dataset.GroupedDataset] constructor.

        Returns:
            A [GroupedDataset][pydvl.valuation.dataset.GroupedDataset] with the initial
                [Dataset][pydvl.valuation.dataset.Dataset] grouped by `data_groups`.
        """
        return cls(
            x=data._x,
            y=data._y,
            data_groups=data_groups,
            feature_names=data.feature_names,
            target_names=data.target_names,
            description=data.description,
            group_names=group_names,
            **kwargs,
        )

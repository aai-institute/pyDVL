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

## NumPy and PyTorch support

It is possible to instantiate [Dataset][pydvl.valuation.dataset.Dataset] and
[GroupedDataset][pydvl.valuation.dataset.GroupedDataset] with either NumPy
arrays or PyTorch tensors. The type of the input arrays is preserved throughout all
operations. Indices are always returned as NumPy arrays, even if the input arrays are
PyTorch tensors.

## Memmapping data

When using numpy arrays for data (and only in that case), it is possible to use
memory mapping to avoid loading the whole dataset into memory, e.g. in subprocesses.
This is done by passing `mmap=True` to the constructor. The data will be stored in a
temporary file, which will be deleted automatically when the creating object is garbage
collected.

If you pass numpy arrays to the constructor, a temporary file will be created and the
data dumped to it, then opened again as a memory map. You must set `mmap=True` even if
the data is already memory-mapped, and in this case no additional shape consistency
checks will be performed, since these might result in new copies. This means that you
might want to call `check_X_y()` manually before constructing the `Dataset`.

See [Working with large datasets][large-datasets-parallelization].

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
    The Dataset and GroupedDataset classes now work with both NumPy arrays and PyTorch
    tensors, preserving the input type throughout operations.
"""

from __future__ import annotations

import logging
import math
import threading
import uuid
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Generic, Sequence, cast, overload

import numpy as np
from deprecate import deprecated
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

__all__ = ["Dataset", "GroupedDataset", "RawData"]

from pydvl.utils.array import (
    Array,
    ArrayT,
    atleast1d,
    check_X_y,
    is_numpy,
    is_tensor,
    stratified_split_indices,
    to_numpy,
    try_torch_import,
)

logger = logging.getLogger(__name__)

# Import torch if available for typing
torch = try_torch_import()
Tensor = None if torch is None else torch.Tensor


@dataclass(frozen=True)
class RawData(Generic[ArrayT]):
    """A view on a dataset's raw data. This is not a copy.

    RawData is a generic container that preserves the type of the arrays it holds.
    It supports both NumPy arrays and PyTorch tensors, but both `x` and `y` must be of
    the same type.
    """

    x: ArrayT
    y: ArrayT

    def __post_init__(self):
        if not (
            (is_numpy(self.x) and is_numpy(self.y))
            or (is_tensor(self.x) and is_tensor(self.y))
        ):
            raise TypeError(
                f"x ({type(self.x)}) and y ({type(self.y)}) must be arrays of the "
                f"same "
                "type (numpy arrays or torch tensors)"
            )

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

    def cpu(self) -> RawData:
        """Returns a new RawData object with tensors moved to the CPU.

        If this `RawData` object hold numpy arrays it returns itself.
        """
        if is_tensor(self.x) and is_tensor(self.y):
            return RawData(self.x.cpu(), self.y.cpu())  # type: ignore
        return self


MMAP_DIR: Path | None = None
_MMAP_LOCK = threading.Lock()


def _cleanup_memmap(path: Path):
    try:
        path.unlink(missing_ok=True)
        logger.info(f"Deleted mmap file '{path}'")
        with _MMAP_LOCK:
            if MMAP_DIR is not None and not MMAP_DIR.exists():
                MMAP_DIR.rmdir()
                logger.info(f"Deleted temporary directory '{MMAP_DIR}'")
    except Exception as e:
        logger.error(f"Error while deleting mmap file '{path}': {e}")


def _maybe_create_memmap(array: NDArray | np.memmap) -> np.memmap:
    global MMAP_DIR
    if isinstance(array, np.memmap):
        return array

    with _MMAP_LOCK:
        if MMAP_DIR is None:
            MMAP_DIR = Path(mkdtemp())
        logger.debug(f"Using temporary directory {MMAP_DIR} for mmap files")

    fname = f"array-{id(array)}-{uuid.uuid4().hex[:6]}.mmap"
    file_path = MMAP_DIR / fname
    mm = np.memmap(file_path, dtype=array.dtype, mode="w+", shape=array.shape)
    logger.debug(f"Created mmap file '{file_path}'")
    mm[:] = array[:]
    mm.flush()
    ro_mm = np.memmap(file_path, dtype=array.dtype, mode="readonly", shape=array.shape)
    weakref.finalize(ro_mm, _cleanup_memmap, file_path)
    return ro_mm


def _maybe_open_mmap(x: Any, dtype: np.dtype, shape: tuple[int, ...]) -> Any:
    if not isinstance(x, Path):
        return x

    return np.memmap(x, dtype=dtype, mode="readonly", shape=shape)


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
        mmap: Whether to use memory-mapped files (for NumPy arrays only). If you pass
            `ndarrays`, a temporary file will be created and the data dumped to it, then
            opened again as a memory map. Use this when working in parallel in a single
            node to avoid unnecessary data copying (at a performance cost). You must set
            `mmap=True` even if the data is already memmapped, and in this case no
            additional shape consistency checks will be performed, meaning you might
            want to call `check_X_y()` manually before memmapping and constructing the
            `Dataset`.

    !!! tip "Type preservation"
        The `Dataset` class preserves the type of input arrays (NumPy or PyTorch)
        throughout all operations. When you access data with slicing or the
        [data()][pydvl.valuation.dataset.Dataset.data] method, the arrays maintain their
        original type.

    !!! tip "Changed in version 0.10.0"
        No longer holds split data, but only `x`, `y`.
        Slicing now return a new `Dataset` object, not raw data.

    !!! tip "New in version 0.11.0"
        Added support for PyTorch tensors.
        Added memory mapping support for numpy arrays.
    """

    feature_names: NDArray[np.str_]
    target_names: NDArray[np.str_]
    _x: ArrayT
    _y: ArrayT
    _indices: NDArray[np.int_]
    _data_names: NDArray[np.str_]

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

            if isinstance(x, np.memmap) and isinstance(y, np.memmap):
                logger.info(
                    "Skipping check_X_y for memmapped arrays. "
                    "Make sure that the data has the proper shape before "
                    "constructing a Dataset"
                )
                _x, _y = x, y
            else:
                _x, _y = check_X_y(x, y, multi_output=multi_output, estimator="Dataset")

            self._x = cast(ArrayT, _maybe_create_memmap(_x))
            self._y = cast(ArrayT, _maybe_create_memmap(_y))
        else:
            self._x, self._y = check_X_y(
                x, y, multi_output=multi_output, estimator="Dataset"
            )

        # These are for __setstate__
        self._x_dtype, self._y_dtype = self._x.dtype, self._y.dtype
        self._x_shape, self._y_shape = self._x.shape, self._y.shape

        def make_names(s: str, a: Array) -> NDArray[np.str_]:
            n = a.shape[1] if len(a.shape) > 1 else 1
            return np.array(
                [f"{s}{i:0{1 + int(math.log10(n))}d}" for i in range(1, n + 1)],
                dtype=str,
            )

        self.feature_names = (
            np.array(feature_names, dtype=str)
            if feature_names is not None
            else make_names("x", x)
        )

        self.target_names = (
            np.array(target_names, dtype=str)
            if target_names is not None
            else make_names("y", y)
        )

        if len(self._x.shape) > 1:
            if len(self.feature_names) != self._x.shape[1]:
                raise ValueError("Mismatching number of features and names")
        if len(self._y.shape) > 1:
            if len(self.target_names) != self._y.shape[1]:
                raise ValueError("Mismatching number of targets and names")

        self.description = description or "No description"
        self._indices = np.arange(len(self._x), dtype=np.int_)
        self._data_names = (
            np.array(data_names, dtype=np.str_)
            if data_names is not None
            else self._indices.astype(np.str_)
        )

    def __getstate__(self):
        """Prepare the object state for pickling replacing memmapped arrays with
        their file paths"""
        state = self.__dict__.copy()

        if isinstance(self._x, np.memmap):
            state["_x"] = Path(self._x.filename)
        if isinstance(self._y, np.memmap):
            state["_y"] = Path(self._y.filename)
        return state

    def __setstate__(self, state):
        """Restore the object state from pickling."""
        self.__dict__.update(state)
        self._x = _maybe_open_mmap(self._x, dtype=self._x_dtype, shape=self._x_shape)
        self._y = _maybe_open_mmap(self._y, dtype=self._y_dtype, shape=self._y_shape)

    def __getitem__(
        self, idx: int | slice | Sequence[int] | NDArray[np.int_] | None = None
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
            feature_idx = np.where(self.feature_names == name)[0][0]
            return np.index_exp[:, feature_idx]  # type: ignore
        except IndexError as e:
            raise ValueError(f"Feature {name} is not in {self.feature_names}") from e

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
        if isinstance(indices, Integral):
            indices = [indices]  # type: ignore
        return RawData(self._x[indices], self._y[indices])

    def data_indices(
        self, indices: Sequence[int] | NDArray[np.int_] | slice | None = None
    ) -> NDArray[np.int_]:
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
        if isinstance(indices, slice):
            return self._indices[indices]
        return cast(NDArray, self._indices[to_numpy(indices)])

    def logical_indices(
        self, indices: Sequence[int] | NDArray[np.int_] | slice | None = None
    ) -> NDArray[np.int_]:
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
        if isinstance(indices, slice):
            return self._indices[indices]
        return cast(NDArray, self._indices[to_numpy(indices)])

    def target(self, name: str) -> tuple[slice, int] | slice:
        """
        Returns a slice or index for the target with the given name.

        If targets are multidimensional (2D array), returns a tuple (slice(None),
        target_idx). If targets are 1D, returns just a slice(None).

        Args:
            name: The name of the target to retrieve.

        Returns:
            For multi-output targets: tuple of (slice(None), target_idx)
            For single-output targets: target_idx (usually 0)

        Raises:
            ValueError: If the target name is not found.
        """
        try:
            target_idx = np.where(self.target_names == name)[0][0]
            if self.n_targets == 1:
                return slice(None)
            else:
                return slice(None), target_idx
        except IndexError as e:
            raise ValueError(f"Target {name} is not in {self.target_names}") from e

    @property
    def n_targets(self) -> int:
        """Returns the number of target variables."""
        return int(self._y.shape[1]) if len(self._y.shape) > 1 else 1

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

    def __repr__(self):
        return (
            f"Dataset({len(self)} samples, {self.n_features} features, "
            f"{self.n_targets} targets, {self.description})"
        )

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
            X: array of shape (n_samples, n_features) - either numpy array or PyTorch
            tensor
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

            x_train, y_train = X[train_indices], y[train_indices]
            x_test, y_test = X[test_indices], y[test_indices]
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                X,
                y,
                train_size=train_size,
                random_state=random_state,
            )

        return cls(x_train, cast(ArrayT, y_train), **kwargs), cls(
            x_test, cast(ArrayT, y_test), **kwargs
        )


class GroupedDataset(Dataset[ArrayT]):
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
            x: training data (numpy array or PyTorch tensor)
            y: labels of training data (same type as x)
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

        !!! note "Type Preservation with Groups"
            GroupedDataset preserves the type of input arrays (NumPy or PyTorch) while
            maintaining all indices as numpy arrays. When accessing grouped data with
            methods like `data()`, the x and y arrays will maintain their original type,
            but indices are always returned as numpy arrays.

        !!! tip "Changed in version 0.6.0"
            Added `group_names` and forwarding of `kwargs`

        !!! tip "Changed in version 0.10.0"
            No longer holds split data, but only x, y and group information. Added
                methods to retrieve indices for groups and vice versa.

        !!! tip "New in version 0.11.0"
            Added support for PyTorch tensors.
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
            {k.item(): [] for k in set(self.data_to_group)}
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

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(
        self, idx: int | slice | Sequence[int] | NDArray[np.int_] | None = None
    ) -> GroupedDataset:
        if idx is None:
            idx = slice(None)
        elif isinstance(idx, int):
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
    def indices(self) -> NDArray[np.int_]:
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
        else:
            indices = to_numpy(indices)
        return np.concatenate([self.group_to_data[i] for i in indices], dtype=np.int_)  # type: ignore

    def logical_indices(
        self, indices: Sequence[int] | NDArray[np.int_] | slice | None = None
    ) -> NDArray[np.int_]:
        """Returns the group indices for the given data indices.

        Args:
            indices: indices of the data points in the data array. If `None`,
                the group indices for all data points are returned.

        Returns:
            Group indices for the given data indices.
        """
        if indices is None:
            return self.data_to_group
        if isinstance(indices, slice):
            return self.data_to_group[indices]
        return cast(NDArray, self.data_to_group[to_numpy(indices)])

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

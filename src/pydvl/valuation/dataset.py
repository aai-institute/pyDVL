"""
This module contains convenience classes to handle data and groups thereof.

Model-based value computations require evaluation of a scoring function (the *utility*).
This is typically the performance of the model on a test set (as an approximation to its
true expected performance). It is therefore convenient to keep both the training data
and the test data grouped to be passed around to methods in [shapley][pydvl.valuation].
This is done with [Dataset][pydvl.valuation.dataset.Dataset].

These underlying data arrays can be accessed via
[Dataset.data][pydvl.valuation.dataset.Dataset.data], which returns the tuple `(X, y)`.
The data can be accessed by indexing the object directly, e.g. `dataset[0]` will return
the data point corresponding to index 0 in `dataset`. Note however that this is not
necessarily the same as `dataset.data().x[0]`, which is the first point in the data
array. This is in particular true for
[GroupedDatasets][pydvl.valuation.dataset.GroupedDataset] where one "logical" index may
correspond to multiple data points. 

Objects of both types can be used to construct [scorers][pydvl.valuation.scorers] and to
fit (most) valuation methods.

## Grouped datasets and logical indices

It is also possible to group data points together with
[GroupedDataset][pydvl.valuation.dataset.dataset.GroupedDataset].

A call to [Dataset.data(indices)][pydvl.valuation.dataset.Dataset.data] will return the
data and labels of all samples for the given groups. But `grouped_data[0]` will return
the data and labels of the first group, not the first data point and will therefore be
in general different from `grouped_data.data([0])`.

In order to handle groups correctly, Datasets map "logical" indices to "data" indices
and vice versa. The latter correspond to indices in the data arrays themselves, while
the former may map to groups of data points.

This is important for valuation methods that require computation on individual data
points, like KNNShapley or Data-OOB. In these cases, the logical indices are used to
compute the Shapley values, while the data indices are used internally by the method.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Iterable, NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch, check_X_y

__all__ = ["Dataset", "GroupedDataset", "RawData"]


logger = logging.getLogger(__name__)


class RawData(NamedTuple):
    x: NDArray
    y: NDArray


class Dataset:
    """A convenience class to handle datasets.

    It holds a dataset, together with info on feature names, target names, and
    data names. It is used to pass data around to valuation methods.

    The underlying data arrays can be accessed via
    [Dataset.data][pydvl.valuation.dataset.Dataset.data], which returns the tuple
    `(X, y)` as a [RawData][pydvl.valuation.dataset.RawData] object. The data can be
    accessed by indexing the object directly, e.g. `dataset[0]` will return the data
    point corresponding to index 0 in `dataset`. For this base class, this is the same
    as `dataset.data([0])`, which is the first point in the data array, but derived
    classes can behave differently.
    """

    _indices: NDArray[np.int_]
    _data_names: NDArray[np.object_]

    def __init__(
        self,
        x: NDArray,
        y: NDArray,
        feature_names: Sequence[str] | None = None,
        target_names: Sequence[str] | None = None,
        data_names: Sequence[str] | None = None,
        description: str | None = None,
        multi_output: bool = False,
    ):
        """Constructs a Dataset from data and labels.

        Args:
            x: training data
            y: labels for training data
            feature_names: names of the features of x data
            target_names: names of the features of y data
            data_names: names assigned to data points.
                For example, if the dataset is a time series, each entry can be a
                timestamp which can be referenced directly instead of using a row
                number.
            description: A textual description of the dataset.
            multi_output: set to `False` if labels are scalars, or to
                `True` if they are vectors of dimension > 1.

        !!! tip "Changed in version 0.10.0"
            No longer holds split data, but only x, y.
        """
        self._x, self._y = check_X_y(
            x, y, multi_output=multi_output, estimator="Dataset"
        )

        def make_names(s: str, a: np.ndarray) -> list[str]:
            n = a.shape[1] if len(a.shape) > 1 else 1
            return [f"{s}{i:0{1 + int(np.log10(n))}d}" for i in range(1, n + 1)]

        self.feature_names = feature_names
        self.target_names = target_names

        if self.feature_names is None:
            self.feature_names = make_names("x", x)

        if self.target_names is None:
            self.target_names = make_names("y", y)

        if len(self._x.shape) > 1:
            if len(self.feature_names) != self._x.shape[-1]:
                raise ValueError("Mismatching number of features and names")
        if len(self._y.shape) > 1:
            if len(self.target_names) != self._y.shape[-1]:
                raise ValueError("Mismatching number of targets and names")

        self.description = description or "No description"
        self._indices = np.arange(len(self._x), dtype=np.int_)
        self._data_names = (
            np.array(data_names, dtype=object)
            if data_names is not None
            else self._indices.astype(object)
        )

    def __getitem__(self, idx: int | slice | Iterable) -> Dataset:
        if isinstance(idx, int):
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
        try:
            return np.index_exp[:, self.feature_names.index(name)]
        except ValueError:
            raise ValueError(f"Feature {name} is not in {self.feature_names}")

    def data(self, indices: Iterable[int] | None = None) -> RawData:
        """Given a set of indices, returns the training data that refer to those
        indices.

        This is used mainly by [Utility][pydvl.valuation.dataset.utility.Utility] to
        retrieve subsets of the data from indices. It is typically **not needed in
        valuation algorithms**.

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

    def data_indices(self, indices: Iterable[int] | None = None) -> NDArray[np.int_]:
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

    def logical_indices(self, indices: Iterable[int] | None = None) -> NDArray[np.int_]:
        """Returns the indices in this Dataset for the given indices in the data array.

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
    def names(self) -> NDArray[np.object_]:
        """Names of each individual datapoint.

        Used for reporting Shapley values.
        """
        return self._data_names

    @property
    def dim(self) -> int:
        """Returns the number of dimensions of a sample."""
        return int(self._x.shape[1]) if len(self._x.shape) > 1 else 1

    def __str__(self):
        return self.description

    def __len__(self):
        return len(self._x)

    @classmethod
    def from_sklearn(
        cls,
        data: Bunch,
        train_size: float = 0.8,
        random_state: int | None = None,
        stratify_by_target: bool = False,
        **kwargs,
    ) -> tuple[Dataset, Dataset]:
        """Constructs two [Dataset][pydvl.valuation.dataset.Dataset] objects from a
        [sklearn.utils.Bunch][], as returned by the `load_*`
        functions in [scikit-learn toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html).

        ??? Example
            ```pycon
            >>> from pydvl.valuation.dataset import Dataset
            >>> from sklearn.datasets import load_boston
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
            random_state: seed for train / test split
            stratify_by_target: If `True`, data is split in a stratified
                fashion, using the target variable as labels. Read more in
                [scikit-learn's user guide](https://scikit-learn.org/stable/modules/cross_validation.html#stratification).
            kwargs: Additional keyword arguments to pass to the
                [Dataset][pydvl.valuation.dataset.Dataset] constructor. Use this to pass e.g. `is_multi_output`.

        Returns:
            Object with the sklearn dataset

        !!! tip "Changed in version 0.6.0"
            Added kwargs to pass to the [Dataset][pydvl.valuation.dataset.Dataset] constructor.
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
        X: NDArray,
        y: NDArray,
        train_size: float = 0.8,
        random_state: int | None = None,
        stratify_by_target: bool = False,
        **kwargs,
    ) -> tuple[Dataset, Dataset]:
        """Constructs a [Dataset][pydvl.valuation.dataset.Dataset] object from X and y numpy arrays  as
        returned by the `make_*` functions in [sklearn generated datasets](https://scikit-learn.org/stable/datasets/sample_generators.html).

        ??? Example
            ```pycon
            >>> from pydvl.valuation.dataset import Dataset
            >>> from sklearn.datasets import make_regression
            >>> X, y = make_regression()
            >>> dataset = Dataset.from_arrays(X, y)
            ```

        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
            train_size: size of the training dataset. Used in `train_test_split`
            random_state: seed for train / test split
            stratify_by_target: If `True`, data is split in a stratified fashion,
                using the y variable as labels. Read more in [sklearn's user
                guide](https://scikit-learn.org/stable/modules/cross_validation.html#stratification).
            kwargs: Additional keyword arguments to pass to the
                [Dataset][pydvl.valuation.dataset.Dataset] constructor. Use this to pass e.g. `feature_names`
                or `target_names`.

        Returns:
            Object with the passed X and y arrays split across training and test sets.

        !!! tip "New in version 0.4.0"

        !!! tip "Changed in version 0.6.0"
            Added kwargs to pass to the [Dataset][pydvl.valuation.dataset.Dataset] constructor.

        !!! tip "Changed in version 0.10.0"
            Returns a tuple of two [Dataset][pydvl.valuation.dataset.Dataset] objects.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            random_state=random_state,
            stratify=y if stratify_by_target else None,
        )
        return cls(x_train, y_train, **kwargs), cls(x_test, y_test, **kwargs)


class GroupedDataset(Dataset):
    def __init__(
        self,
        x: NDArray,
        y: NDArray,
        data_groups: Sequence[int],
        feature_names: Sequence[str] | None = None,
        target_names: Sequence[str] | None = None,
        data_names: Sequence[str] | None = None,
        group_names: Sequence[str] | None = None,
        description: str | None = None,
        **kwargs,
    ):
        """Class for grouping datasets.

        Used for calculating values of subsets of the data considered as logical units.
        For instance, one can group by value of a categorical feature, by bin into which
        a continuous feature falls, or by label.

        Args:
            x: training data
            y: labels of training data
            data_groups: Iterable of the same length as `x_train` containing
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
                methods to retrieve indices for groups and vicecersa.
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
        self.data_to_group = np.array(data_groups, dtype=int)
        # abstract index (group id) -> data index
        self.group_to_data: OrderedDict[int, list[int]] = OrderedDict(
            {k: [] for k in set(data_groups)}
        )
        for data_idx, group_idx in enumerate(self.data_to_group):
            self.group_to_data[group_idx].append(data_idx)  # type: ignore
        self._indices = np.array(list(self.group_to_data.keys()))
        self._group_names = (
            np.array(group_names, dtype=object)
            if group_names is not None
            else np.array(list(self.group_to_data.keys()), dtype=object)
        )

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx: int | slice | Iterable) -> GroupedDataset:
        if isinstance(idx, int):
            idx = [idx]
        return GroupedDataset(
            x=self._x[self.data_indices(idx)],
            y=self._y[self.data_indices(idx)],
            data_groups=self.data_to_group[self.data_indices(idx)],
            feature_names=self.feature_names,
            target_names=self.target_names,
            data_names=self._data_names[self.data_indices(idx)],
            group_names=self._group_names[idx],
            description="(SLICED): " + self.description,
        )

    @property
    def indices(self):
        """Indices of the groups."""
        return self._indices

    @property
    def names(self) -> NDArray[object]:
        """Names of the groups."""
        # FIXME? this shadows _data_names (but it can still be accessed...)
        return self._group_names

    def data(self, indices: Iterable[int] | None = None) -> RawData:
        """Returns the data and labels of all samples in the given groups.

        Args:
            indices: group indices whose elements to return. If `None`,
                all data from all groups are returned.

        Returns:
            Tuple of training data `x` and labels `y`.
        """
        return super().data(self.data_indices(indices))

    def data_indices(self, indices: Iterable[int] | None = None) -> NDArray[np.int_]:
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

    def logical_indices(self, indices: Iterable[int] | None = None) -> NDArray[np.int_]:
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

    @classmethod
    def from_sklearn(
        cls,
        data: Bunch,
        train_size: float = 0.8,
        stratify_by_target: bool = False,
        data_groups: Sequence | None = None,
        random_state: int | None = None,
        **kwargs,
    ) -> tuple[GroupedDataset, GroupedDataset]:
        """Constructs a [GroupedDataset][pydvl.valuation.dataset.GroupedDataset] object, and an
        ungrouped [Dataset][pydvl.valuation.dataset.Dataset] object from a
        [sklearn.utils.Bunch][sklearn.utils.Bunch] as returned by the `load_*` functions in
        [scikit-learn toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html) and groups
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
            train_size: size of the training dataset. Used in `train_test_split`.
            random_state: seed for train / test split.
            stratify_by_target: If `True`, data is split in a stratified
                fashion, using the target variable as labels. Read more in
                [sklearn's user guide](https://scikit-learn.org/stable/modules/cross_validation.html#stratification).
            data_groups: an array holding the group index or name for each
                data point. The length of this array must be equal to the number of
                data points in the dataset.
            kwargs: Additional keyword arguments to pass to the
                [Dataset][pydvl.valuation.dataset.Dataset] constructor.

        Returns:
            Datasets with the selected sklearn data

        !!! tip "Changed in version 0.10.0"
            Returns a tuple of two [GroupedDataset][pydvl.valuation.dataset.GroupedDataset]
                objects.
        """
        return cls.from_arrays(
            X=data.data,
            y=data.target,
            train_size=train_size,
            stratify_by_target=stratify_by_target,
            data_groups=data_groups,
            random_state=random_state,
            **kwargs,
        )

    @classmethod
    def from_arrays(
        cls,
        X: NDArray,
        y: NDArray,
        train_size: float = 0.8,
        stratify_by_target: bool = False,
        data_groups: Sequence | None = None,
        random_state: int | None = None,
        **kwargs,
    ) -> tuple[GroupedDataset, GroupedDataset]:
        """Constructs a [GroupedDataset][pydvl.valuation.dataset.GroupedDataset] object,
        and an ungrouped [Dataset][pydvl.valuation.dataset.Dataset] object from X and y
        numpy arrays as returned by the `make_*` functions in
        [scikit-learn generated datasets](https://scikit-learn.org/stable/datasets/sample_generators.html).

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
                [sklearn's user guide](https://scikit-learn.org/stable/modules/cross_validation.html#stratification).
            data_groups: an array holding the group index or name for each data
                point. The length of this array must be equal to the number of
                data points in the dataset.
            kwargs: Additional keyword arguments that will be passed to the
                [Dataset][pydvl.valuation.dataset.Dataset] constructor.

        Returns:
            Dataset with the passed X and y arrays split across training and
                test sets.

        !!! tip "New in version 0.4.0"

        !!! tip "Changed in version 0.6.0"
            Added kwargs to pass to the [Dataset][pydvl.valuation.dataset.Dataset]
                constructor.

        !!! tip "Changed in version 0.10.0"
            Returns a tuple of two [GroupedDataset][pydvl.valuation.dataset.GroupedDataset]
                objects.
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
    def from_dataset(cls, data: Dataset, data_groups: Sequence[Any]) -> GroupedDataset:
        """Creates a [GroupedDataset][pydvl.valuation.dataset.GroupedDataset] object from a
        [Dataset][pydvl.valuation.dataset.Dataset] object and a mapping of data groups.

        ??? Example
            ```pycon
            >>> import numpy as np
            >>> from pydvl.valuation.dataset import Dataset, GroupedDataset
            >>> train, test = Dataset.from_arrays(
            ...     X=np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]]),
            ...     y=np.asarray([0, 1, 0, 1]),
            ... )
            >>> grouped_train = GroupedDataset.from_dataset(train, data_groups=[0, 0, 1, 1])
            ```

        Args:
            data: The original data.
            data_groups: An array holding the group index or name for each data
                point. The length of this array must be equal to the number of
                data points in the dataset.

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
        )

"""
This module contains convenience classes to handle data and groups thereof.

Model-based value computations require evaluation of a scoring function (the *utility*).
This is typically the performance of the model on a test set (as an approximation to its
true expected performance). It is therefore convenient to keep both the training data
and the test data grouped to be passed around to methods in [shapley][pydvl.valuation].
This is done with [Dataset][pydvl.valuation.dataset.Dataset].

This abstraction layer also seamlessly groups data points together if one is interested
in computing their value as a group, see
[GroupedDataset][pydvl.valuation.dataset.dataset.GroupedDataset].

Objects of both types can be used to construct [scorers][pydvl.valuation.scorers] and to
fit valuation methods.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch, check_X_y

__all__ = ["Dataset", "GroupedDataset"]

logger = logging.getLogger(__name__)


class Dataset:
    """A convenience class to handle datasets.

    It holds a dataset, together with info on feature names, target names, and
    data names. It is used to pass data around to valuation methods.
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
        self.x, self.y = check_X_y(x, y, multi_output=multi_output, estimator="Dataset")

        def make_names(s: str, a: np.ndarray) -> list[str]:
            n = a.shape[1] if len(a.shape) > 1 else 1
            return [f"{s}{i:0{1 + int(np.log10(n))}d}" for i in range(1, n + 1)]

        self.feature_names = feature_names
        self.target_names = target_names

        if self.feature_names is None:
            self.feature_names = make_names("x", x)

        if self.target_names is None:
            self.target_names = make_names("y", y)

        if len(self.x.shape) > 1:
            if len(self.feature_names) != self.x.shape[-1]:
                raise ValueError("Mismatching number of features and names")
        if len(self.y.shape) > 1:
            if len(self.target_names) != self.y.shape[-1]:
                raise ValueError("Mismatching number of targets and names")

        self.description = description or "No description"
        self._indices = np.arange(len(self.x), dtype=np.int_)
        self._data_names = (
            np.array(data_names, dtype=object)
            if data_names is not None
            else self._indices.astype(object)
        )

    def __getitem__(self, idx: int | slice | Iterable) -> tuple:
        return self.x[idx], self.y[idx]

    def feature(self, name: str) -> tuple[slice, int]:
        try:
            return np.index_exp[:, self.feature_names.index(name)]  # type: ignore
        except ValueError:
            raise ValueError(f"Feature {name} is not in {self.feature_names}")

    def get_data(self, indices: Iterable[int] | None = None) -> tuple[NDArray, NDArray]:
        """Given a set of indices, returns the training data that refer to those
        indices.

        This is used mainly by [Utility][pydvl.valuation.dataset.utility.Utility] to retrieve
        subsets of the data from indices. It is typically **not needed in
        algorithms**.

        Args:
            indices: Optional indices that will be used to select points from
                the training data. If `None`, the entire training data will be
                returned.

        Returns:
            If `indices` is not `None`, the selected x and y arrays from the
                training data. Otherwise, the entire dataset.
        """
        if indices is None:
            return self.x, self.y
        return self.x[indices], self.y[indices]

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
    def data_names(self) -> NDArray[np.object_]:
        """Names of each individual datapoint.

        Used for reporting Shapley values.
        """
        return self._data_names

    @property
    def dim(self) -> int:
        """Returns the number of dimensions of a sample."""
        return int(self.x.shape[1]) if len(self.x.shape) > 1 else 1

    def __str__(self):
        return self.description

    def __len__(self):
        return len(self.x)

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
        data_groups: Sequence,
        feature_names: Sequence[str] | None = None,
        target_names: Sequence[str] | None = None,
        group_names: Sequence[str] | None = None,
        description: str | None = None,
        **kwargs,
    ):
        """Class for grouping datasets.

        Used for calculating Shapley values of subsets of the data considered
        as logical units. For instance, one can group by value of a categorical
        feature, by bin into which a continuous feature falls, or by label.

        Args:
            x: training data
            y: labels of training data
            data_groups: Iterable of the same length as `x_train` containing
                a group label for each training data point. The label can be of any
                type, e.g. `str` or `int`. Data points with the same label will
                then be grouped by this object and considered as one for effects of
                valuation.
            feature_names: names of the covariates' features.
            target_names: names of the labels or targets y
            group_names: names of the groups. If not provided, the labels
                from `data_groups` will be used.
            description: A textual description of the dataset
            kwargs: Additional keyword arguments to pass to the
                [Dataset][pydvl.valuation.dataset.Dataset] constructor.

        !!! tip "Changed in version 0.6.0"
            Added `group_names` and forwarding of `kwargs`

        !!! tip "Changed in version 0.10.0"
            No longer holds split data, but only x,y and group information.
        """
        super().__init__(
            x=x,
            y=y,
            feature_names=feature_names,
            target_names=target_names,
            description=description,
            **kwargs,
        )

        if len(data_groups) != len(x):
            raise ValueError(
                f"data_groups and x_train must have the same length."
                f"Instead got {len(data_groups)=} and {len(x)=}"
            )

        self.groups: OrderedDict[Any, list[int]] = OrderedDict(
            {k: [] for k in set(data_groups)}
        )
        for idx, group in enumerate(data_groups):
            self.groups[group].append(idx)
        self.group_items = list(self.groups.items())
        self._indices = np.arange(len(self.groups.keys()))
        self._data_names = (
            np.array(group_names, dtype=object)
            if group_names is not None
            else np.array(list(self.groups.keys()), dtype=object)
        )

    def __len__(self):
        return len(self.groups)

    @property
    def indices(self):
        """Indices of the groups."""
        return self._indices

    # FIXME this is a misnomer, should be `names` in `Dataset` so that here it
    #  makes sense
    @property
    def data_names(self):
        """Names of the groups."""
        return self._data_names

    def get_data(self, indices: Iterable[int] | None = None) -> tuple[NDArray, NDArray]:
        """Returns the data and labels of all samples in the given groups.

        Args:
            indices: group indices whose elements to return. If `None`,
                all data from all groups are returned.

        Returns:
            Tuple of training data x and labels y.
        """
        if indices is None:
            indices = self.indices
        data_indices = [
            idx for group_id in indices for idx in self.group_items[group_id][1]
        ]
        return super().get_data(data_indices)

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
            x=data.x,
            y=data.y,
            data_groups=data_groups,
            feature_names=data.feature_names,
            target_names=data.target_names,
            description=data.description,
        )

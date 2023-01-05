"""
This module contains convenience classes to handle data and groups thereof.

Shapley and Least Core value computations require evaluation of a scoring function
(the *utility*). This is typically the performance of the model on a test set
(as an approximation to its true expected performance). It is therefore convenient
to keep both the training data and the test data together to be passed around to
methods in :mod:`~pydvl.value.shapley` and :mod:`~pydvl.value.least_core`.
This is done with :class:`~pydvl.utils.dataset.Dataset`.

This abstraction layer also seamlessly grouping data points together if one is
interested in computing their value as a group, see
:class:`~pydvl.utils.dataset.GroupedDataset`.

Objects of both types are used to construct a :class:`~pydvl.utils.utility.Utility`
object.

"""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import Bunch, check_X_y

__all__ = ["Dataset", "GroupedDataset", "load_spotify_dataset", "load_wine_dataset"]

logger = logging.getLogger(__name__)


class Dataset:
    """A convenience class to handle datasets.

    It holds a dataset, split into training and test data, together with several
    labels on feature names, data point names and a description.
    """

    def __init__(
        self,
        x_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.DataFrame],
        x_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[Sequence[str]] = None,
        target_names: Optional[Sequence[str]] = None,
        data_names: Optional[Sequence[str]] = None,
        description: Optional[str] = None,
        is_multi_output: bool = False,
    ):
        """Constructs a Dataset from data and labels.

        :param x_train: training data
        :param y_train: labels for training data
        :param x_test: test data
        :param y_test: labels for test data
        :param feature_names: name of the features of input data
        :param target_names: names of the features of target data
        :param data_names: names assigned to data points.
            For example, if the dataset is a time series, each entry can be a
            timestamp which can be referenced directly instead of using a row
            number.
        :param description: A textual description of the dataset.
        :param is_multi_output: set to True if y holds multiple labels for each
            data point. False if y is a 1d array holding a single label per
            point.
        """
        self.x_train, self.y_train = check_X_y(
            x_train, y_train, multi_output=is_multi_output
        )
        self.x_test, self.y_test = check_X_y(
            x_test, y_test, multi_output=is_multi_output
        )

        if x_train.shape[-1] != x_test.shape[-1]:
            raise ValueError(
                f"Mismatching number of features: "
                f"{x_train.shape[-1]} and {x_test.shape[-1]}"
            )
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"Mismatching number of samples: "
                f"{x_train.shape[-1]} and {x_test.shape[-1]}"
            )
        if x_test.shape[0] != y_test.shape[0]:
            raise ValueError(
                f"Mismatching number of samples: "
                f"{x_test.shape[-1]} and {y_test.shape[-1]}"
            )

        def make_names(s: str, a: np.ndarray) -> List[str]:
            n = a.shape[1] if len(a.shape) > 1 else 1
            return [f"{s}{i:0{1 + int(np.log10(n))}d}" for i in range(1, n + 1)]

        self.feature_names = feature_names
        self.target_names = target_names

        if self.feature_names is None:
            if isinstance(x_train, pd.DataFrame):
                self.feature_names = x_train.columns.tolist()
            else:
                self.feature_names = make_names("x", x_train)

        if self.target_names is None:
            if isinstance(y_train, pd.DataFrame):
                self.target_names = y_train.columns.tolist()
            else:
                self.target_names = make_names("y", y_train)

        if len(self.x_train.shape) > 1:
            if (
                len(self.feature_names) != self.x_train.shape[-1]
                or len(self.feature_names) != self.x_test.shape[-1]
            ):
                raise ValueError("Mismatching number of features and names")
        if len(self.y_train.shape) > 1:
            if (
                len(self.target_names) != self.y_train.shape[-1]
                or len(self.target_names) != self.y_test.shape[-1]
            ):
                raise ValueError("Mismatching number of targets and names")

        self.description = description or "No description"
        self._indices = np.arange(len(self.x_train))
        self._data_names = data_names if data_names is not None else self._indices

    def __iter__(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def __getitem__(self, idx: Union[int, slice, Iterable]) -> Tuple:
        return self.x_train[idx], self.y_train[idx]

    def feature(self, name: str) -> Tuple[slice, int]:
        try:
            return np.index_exp[:, self.feature_names.index(name)]  # type: ignore
        except ValueError:
            raise ValueError(f"Feature {name} is not in {self.feature_names}")

    def get_training_data(
        self, indices: Optional[Iterable[int]] = None
    ) -> Tuple[NDArray, NDArray]:
        """Given a set of indices, returns the training data that refer to those
        indices.

        This is used when calling different sub-sets of indices to calculate
        shapley values. Notice that train_indices is not typically equal to the
        full indices, but only a subset of it.

        :param indices: Optional indices that will be used
            to select data points from the training data.
        :return: If indices is not None, the selected x and y arrays from
            the training data. Otherwise, the entire training data.
        """
        if indices is None:
            return self.x_train, self.y_train
        x = self.x_train[indices]
        y = self.y_train[indices]
        return x, y

    def get_test_data(
        self, indices: Optional[Iterable[int]] = None
    ) -> Tuple[NDArray, NDArray]:
        """Returns the entire test set regardless of the passed indices.

        The passed indices will not be used because for data valuation
        we generally want to score the trained model on the entire test data.

        Additionally, the way this method is used in the
        :class:`~pydvl.utils.utility.Utility` class, the passed indices will
        be those of the training data and would not work on the test data.

        There may be cases where it is desired to use parts of the test data.
        In those cases, it is recommended to inherit from the :class:`Dataset`
        class and to override the :meth:`~Dataset.get_test_data` method.

        For example, the following snippet shows how one could go about
        mapping the training data indices into test data indices
        inside :meth:`~Dataset.get_test_data`:

        :Example:

            >>> from pydvl.utils import Dataset
            >>> import numpy as np
            >>> class DatasetWithTestDataIndices(Dataset):
            ...    def get_test_data(self, indices=None):
            ...        if indices is None:
            ...            return self.x_test, self.y_test
            ...        fraction = len(list(indices)) / len(self)
            ...        mapped_indices = len(self.x_test) / len(self) * np.asarray(indices)
            ...        mapped_indices = np.unique(mapped_indices.astype(int))
            ...        return self.x_test[mapped_indices], self.y_test[mapped_indices]
            ...
            >>> X = np.random.rand(100, 10)
            >>> y = np.random.randint(0, 2, 100)
            >>> dataset = DatasetWithTestDataIndices.from_arrays(X, y)
            >>> indices = np.random.choice(dataset.indices, 30, replace=False)
            >>> _ = dataset.get_training_data(indices)
            >>> _ = dataset.get_test_data(indices)


        :param indices: Optional indices into the test data. This argument
            is unused and is left as is to keep the same interface as
            :meth:`Dataset.get_training_data`.
        :return: The entire test data.
        """
        return self.x_test, self.y_test

    def target(self, name: str) -> Tuple[slice, int]:
        try:
            return np.index_exp[:, self.target_names.index(name)]  # type: ignore
        except ValueError:
            raise ValueError(f"Target {name} is not in {self.target_names}")

    @property
    def indices(self):
        """Index of positions in data.x_train.

        Contiguous integers from 0 to len(Dataset).
        """
        return self._indices

    @property
    def data_names(self):
        """Names of each individual datapoint.

        Used for reporting Shapley values.
        """
        return self._data_names

    @property
    def dim(self):
        """Returns the number of dimensions of a sample."""
        return self.x_train.shape[1] if len(self.x_train.shape) > 1 else 1

    def __str__(self):
        return self.description

    def __len__(self):
        return len(self.x_train)

    @classmethod
    def from_sklearn(
        cls,
        data: Bunch,
        train_size: float = 0.8,
        random_state: Optional[int] = None,
        stratify_by_target: bool = False,
    ) -> "Dataset":
        """Constructs a :class:`Dataset` object from an :class:`sklearn.utils.Bunch` bunch as returned by the
        `load_*` functions in `sklearn toy datasets
        <https://scikit-learn.org/stable/datasets/toy_dataset.html>`_.

        :param data: sklearn dataset
        :param train_size: size of the training dataset. Used in
            `train_test_split`
        :param random_state: seed for train / test split
        :param stratify_by_target: If `True`, data is split in a stratified
            fashion, using the target variable as labels. Read more in
            `sklearn's user guide
            <https://scikit-learn.org/stable/modules/cross_validation.html
            #stratification>`.

        :return: Dataset with the selected sklearn data
        """
        x_train, x_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            train_size=train_size,
            random_state=random_state,
            stratify=data.target if stratify_by_target else None,
        )
        return cls(
            x_train,
            y_train,
            x_test,
            y_test,
            feature_names=data.get("feature_names"),
            target_names=data.get("target_names"),
            description=data.get("DESCR"),
        )

    @classmethod
    def from_arrays(
        cls,
        X: NDArray,
        y: NDArray,
        train_size: float = 0.8,
        random_state: Optional[int] = None,
        stratify_by_target: bool = False,
    ) -> "Dataset":
        """.. versionadded:: 0.4.0

        Constructs a :class:`Dataset` object from X and y numpy arrays  as returned by the
        `make_*` functions in `sklearn generated datasets
        <https://scikit-learn.org/stable/datasets/sample_generators.html>`_.

        :param X: numpy array of shape (n_samples, n_features)
        :param y: numpy array of shape (n_samples,)
        :param train_size: size of the training dataset. Used in
            `train_test_split`
        :param random_state: seed for train / test split
        :param stratify_by_target: If `True`, data is split in a stratified
            fashion, using the y variable as labels. Read more in
            `sklearn's user guide
            <https://scikit-learn.org/stable/modules/cross_validation.html
            #stratification>`.

        :return: Dataset with the passed X and y arrays split across training and test sets.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_size,
            random_state=random_state,
            stratify=y if stratify_by_target else None,
        )
        return cls(
            x_train,
            y_train,
            x_test,
            y_test,
        )


class GroupedDataset(Dataset):
    """Class that groups data points.

    Used for calculating Shapley values of coalitions.
    """

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        data_groups: Sequence,
        feature_names: Optional[Sequence[str]] = None,
        target_names: Optional[Sequence[str]] = None,
        description: Optional[str] = None,
    ):
        """Class for grouping datasets.

        :param x_train: training data
        :param y_train: labels of training data
        :param x_test: input of test data
        :param y_test: labels of test data
        :param data_groups: Iterable of the same length as `x_train` containing
            a group label for each training data point. The label can be of any
            type, e.g. `str` or `int`. Data points with the same label will then
            be grouped by this object and considered as one for effects of
            valuation.
        :param feature_names: name of the features of input data
        :param target_names: name of target data
        :param description: description of the dataset
        """
        super().__init__(
            x_train, y_train, x_test, y_test, feature_names, target_names, description
        )
        if len(data_groups) != len(x_train):
            raise ValueError(
                f"data_groups and x_train must have the same length."
                f"Instead got {len(data_groups)=} and {len(x_train)=}"
            )

        self.groups: OrderedDict[Any, List[int]] = OrderedDict(
            {k: [] for k in set(data_groups)}
        )
        for idx, group in enumerate(data_groups):
            self.groups[group].append(idx)
        self.group_items = list(self.groups.items())
        self._indices = np.arange(len(self.groups.keys()))

    def __len__(self):
        return len(self.groups)

    @property
    def indices(self):
        """Indices of the grouped data points.

        These are not the indices of all the dataset, but only those referencing
        the groups.
        """
        return np.array(self._indices)

    @property
    def data_names(self):
        """Name given to the groups."""
        return list(self.groups.keys())

    def get_training_data(self, train_indices):
        """Given a set of indices, it returns the related groups."""
        data_indices = [
            idx for group_id in train_indices for idx in self.group_items[group_id][1]
        ]
        return super().get_training_data(data_indices)

    @classmethod
    def from_sklearn(
        cls,
        data: Bunch,
        train_size: float = 0.8,
        random_state: Optional[int] = None,
        stratify_by_target: bool = False,
        data_groups: Optional[List] = None,
    ) -> "GroupedDataset":
        """Constructs a :class:`GroupedDataset` object from an sklearn bunch as returned by the
        `load_*` functions in `sklearn toy datasets
        <https://scikit-learn.org/stable/datasets/toy_dataset.html>`_ and groups
        it.

        :param data: sklearn dataset
        :param train_size: size of the training dataset. Used in
            `train_test_split`.
        :param random_state: seed for train / test split.
        :param stratify_by_target: If `True`, data is split in a stratified
            fashion, using the target variable as labels. Read more in
            `sklearn's user guide
            <https://scikit-learn.org/stable/modules/cross_validation.html
            #stratification>`.
        :param data_groups: for each element in the training set, it associates
            a group index or name.

        :return: Dataset with the selected sklearn data
        """
        if data_groups is None:
            raise ValueError("data_groups argument is missing")
        dataset = Dataset.from_sklearn(
            data, train_size, random_state, stratify_by_target
        )
        return cls.from_dataset(dataset, data_groups)

    @classmethod
    def from_arrays(
        cls,
        X: NDArray,
        y: NDArray,
        train_size: float = 0.8,
        random_state: Optional[int] = None,
        stratify_by_target: bool = False,
        data_groups: Optional[List] = None,
    ) -> "Dataset":
        """.. versionadded:: 0.4.0

        Constructs a :class:`GroupedDataset` object from X and y numpy arrays  as returned by the
        `make_*` functions in `sklearn generated datasets
        <https://scikit-learn.org/stable/datasets/sample_generators.html>`_.

        :param X: numpy array of shape (n_samples, n_features)
        :param y: numpy array of shape (n_samples,)
        :param train_size: size of the training dataset. Used in
            `train_test_split`
        :param random_state: seed for train / test split
        :param stratify_by_target: If `True`, data is split in a stratified
            fashion, using the y variable as labels. Read more in
            `sklearn's user guide
            <https://scikit-learn.org/stable/modules/cross_validation.html
            #stratification>`.
        :param data_groups: for each element in the training set, it associates
            a group index or name.

        :return: Dataset with the passed X and y arrays split across training and test sets.
        """
        if data_groups is None:
            raise ValueError("data_groups argument is missing")
        dataset = Dataset.from_arrays(
            X, y, train_size, random_state, stratify_by_target
        )
        return cls.from_dataset(dataset, data_groups)

    @classmethod
    def from_dataset(
        cls, dataset: Dataset, data_groups: Sequence[Any]
    ) -> "GroupedDataset":
        """Given a :class:`Dataset` object, it creates it into a :class:`GroupedDataset` object
        by passing a list of data groups, one for each element in the training set.

        :param dataset: :class:`Dataset` object
        :param data_groups: for each element in the training set, it associates a group
            index or name.
        :return: :class:`GroupedDataset`, with the initial :class:`Dataset` grouped by data_groups.
        """
        return cls(
            x_train=dataset.x_train,
            y_train=dataset.y_train,
            x_test=dataset.x_test,
            y_test=dataset.y_test,
            data_groups=data_groups,
            feature_names=dataset.feature_names,
            target_names=dataset.target_names,
            description=dataset.description,
        )


def load_spotify_dataset(
    val_size: float,
    test_size: float,
    min_year: int = 2014,
    target_column: str = "popularity",
    random_state: int = 24,
):
    """Loads (and downloads if not already cached) the spotify music dataset.
    More info on the dataset can be found at
    https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify.

    If this method is called within the CI pipeline, it will load a reduced
    version of the dataset for testing purposes.

    :param val_size: size of the validation set
    :param test_size: size of the test set
    :param min_year: minimum year of the returned data
    :param target_column: column to be returned as y (labels)
    :param random_state: fixes sklearn random seed
    :return: Tuple with 3 elements, each being a list sith [input_data, related_labels]
    """
    root_dir_path = Path(__file__).parent.parent.parent.parent
    file_path = root_dir_path / "data/top_hits_spotify_dataset.csv"
    if file_path.exists():
        data = pd.read_csv(file_path)
    else:
        url = "https://github.com/appliedAI-Initiative/pyDVL/blob/develop/data/top_hits_spotify_dataset.csv"
        data = pd.read_csv(url)
        data.to_csv(file_path, index=False)

    data = data[data["year"] > min_year]
    data["genre"] = data["genre"].astype("category").cat.codes
    y = data[target_column]
    X = data.drop(target_column, axis=1)
    X, X_test, y, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state
    )
    return [X_train, y_train], [X_val, y_val], [X_test, y_test]


def load_wine_dataset(
    train_size: float, test_size: float, random_state: Optional[int] = None
):
    """Loads the sklearn wine dataset. More info can be found at
    https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset.

    :param train_size: fraction of points used for training dataset
    :param test_size: fraction of points used for test dataset
    :param random_state: fix random seed. If None, no random seed is set.
    :return: A tuple of four elements with the first three being input and
        target values in the form of matrices of shape (N,D) the first
        and (N,) the second. The fourth element is a list containing names of
        features of the model. (FIXME doc)
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "PyTorch is required in order to load the Wine Dataset"
        ) from e

    wine_bunch = load_wine(as_frame=True)
    x, x_test, y, y_test = train_test_split(
        wine_bunch.data,
        wine_bunch.target,
        train_size=1 - test_size,
        random_state=random_state,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, train_size=train_size / (1 - test_size), random_state=random_state
    )
    x_transformer = MinMaxScaler()

    transformed_x_train = x_transformer.fit_transform(x_train)
    transformed_x_test = x_transformer.transform(x_test)

    transformed_x_train = torch.tensor(transformed_x_train, dtype=torch.float)
    transformed_y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)

    transformed_x_test = torch.tensor(transformed_x_test, dtype=torch.float)
    transformed_y_test = torch.tensor(y_test.to_numpy(), dtype=torch.long)

    transformed_x_val = x_transformer.transform(x_val)
    transformed_x_val = torch.tensor(transformed_x_val, dtype=torch.float)
    transformed_y_val = torch.tensor(y_val.to_numpy(), dtype=torch.long)
    return (
        (transformed_x_train, transformed_y_train),
        (transformed_x_val, transformed_y_val),
        (transformed_x_test, transformed_y_test),
        wine_bunch.feature_names,
    )


def synthetic_classification_dataset(
    mus: np.ndarray,
    sigma: float,
    num_samples: int,
    train_size: float,
    test_size: float,
    random_seed=None,
) -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Tuple[Any, Any]]:
    """Sample from a uniform Gaussian mixture model.

    :param mus: 2d-matrix [CxD] with the means of the components in the rows.
    :param sigma: Standard deviation of each dimension of each component.
    :param num_samples: The number of samples to generate.
    :param train_size: fraction of points used for training dataset
    :param test_size: fraction of points used for test dataset
    :param random_seed: fix random seed. If None, no random seed is set.
    :returns: A tuple of matrix x of shape [NxD] and target vector y of shape [N].
    """
    num_features = mus.shape[1]
    num_classes = mus.shape[0]
    gaussian_cov = sigma * np.eye(num_features)
    gaussian_chol = np.linalg.cholesky(gaussian_cov)
    y = np.random.randint(num_classes, size=num_samples)
    x = (
        np.einsum(
            "ij,kj->ki",
            gaussian_chol,
            np.random.normal(size=[num_samples, num_features]),
        )
        + mus[y]
    )
    x, x_test, y, y_test = train_test_split(
        x,
        y,
        train_size=1 - test_size,
        random_state=random_seed,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, train_size=train_size / (1 - test_size), random_state=random_seed
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def decision_boundary_fixed_variance_2d(
    mu_1: np.ndarray, mu_2: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Closed-form solution for decision boundary dot(a, b) + b = 0 with fixed variance.
    :param mu_1: First mean.
    :param mu_2: Second mean.
    :returns: A callable which converts a continuous line (-infty, infty) to the decision boundary in feature space.
    """
    a = np.asarray([[0, 1], [-1, 0]]) @ (mu_2 - mu_1)
    b = (mu_1 + mu_2) / 2
    a = a.reshape([1, -1])
    return lambda z: z.reshape([-1, 1]) * a + b  # type: ignore

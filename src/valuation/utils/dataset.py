import os
from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import Bunch, check_X_y

__all__ = ["Dataset", "GroupedDataset", "load_spotify_dataset", "load_wine_dataset"]


class Dataset:
    """Class for better handling datasets"""

    def __init__(
        self,
        x_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.DataFrame],
        x_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[Sized] = None,
        target_names: Optional[Sized] = None,
        data_names: Optional[Sized] = None,
        description: str = None,
        is_multi_output=False,
    ):
        """It holds a dataset, split into train and test data, together
        with several labels on feature names, data point names and description

        :param x_train: train input data
        :param y_train: labels of train data
        :param x_test: input of test data
        :param y_test: labels of test data
        :param feature_names: name of the features of input data
        :param target_names: name of target data
        :param data_names: name given to each data point.
            For example, if the dataset is a time series, each row represents
            a time step which can be referenced directly using timestamps instead
            of the row number.
        :param is_multi_output: set to True if y holds multiple labels for each data point.
            False if y is a 1d array holding a single label per point.
        :param description: description of the dataset
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

        def make_names(s: str, a: np.ndarray) -> np.ndarray:
            n = a.shape[1] if len(a.shape) > 1 else 1
            return np.array(
                [f"{s}{i:0{1 + int(np.log10(n))}d}" for i in range(1, n + 1)]
            )

        self.feature_names = feature_names
        self.target_names = target_names

        if self.feature_names is None:
            if isinstance(x_train, pd.DataFrame):
                self.feature_names = np.asarray(x_train.columns)
            else:
                self.feature_names = make_names("x", x_train)

        if self.target_names is None:
            if isinstance(y_train, pd.DataFrame):
                self.target_names = np.asarray(y_train.columns)
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

    def feature(self, name: str) -> Tuple[slice, int]:
        try:
            return np.index_exp[:, self.feature_names.index(name)]  # type: ignore
        except ValueError:
            raise ValueError(f"Feature {name} is not in {self.feature_names}")

    def get_train_data(self, train_indices: Optional[Iterable[int]]):
        """Given a set of indices, it returns the train data that refer to those indices.
        This is used when calling different sub-sets of indices to calculate shapley values.
        Notice that train_indices is not typically equal to the full indices, but only a subset of it.
        """
        if train_indices is None:
            return self.x_train, self.y_train
        else:
            x = self.x_train[train_indices]
            y = self.y_train[train_indices]
            return x, y

    def get_test_data(self, test_indices: Optional[Iterable[int]]):
        """Given a set of indices, it returns the test data that refer to those indices."""
        if test_indices is None:
            return self.x_test, self.y_test
        else:
            x = self.x_test[test_indices]
            y = self.y_test[test_indices]
            return x, y

    def target(self, name: str) -> Tuple[slice, int]:
        try:
            return np.index_exp[:, self.target_names.index(name)]  # type: ignore
        except ValueError:
            raise ValueError(f"Target {name} is not in {self.target_names}")

    @property
    def indices(self):
        """Index of positions in data.x_train. Contiguous integers from 0 to
        len(Dataset)."""
        return self._indices

    @property
    def data_names(self):
        """Names of each individual datapoint. Used for reporting Shapley values."""
        return self._data_names

    @property
    def dim(self):
        """Returns the number of dimensions of a sample."""
        return self.x_train.shape[1] if len(self.x_train.shape) > 1 else 1

    # hacky 🙈
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
    ) -> "Dataset":
        """Constructs a Dataset object from an sklearn bunch as returned
        by the load_* functions in `sklearn.datasets`.

        :param data: sklearn dataset
        :param train_size: size of the training dataset. Used in train_test_split
        :param random_state: seed for train test split
        :return: Dataset with the selected sklearn data
        """
        x_train, x_test, y_train, y_test = train_test_split(
            data.data, data.target, train_size=train_size, random_state=random_state
        )
        return Dataset(
            x_train,
            y_train,
            x_test,
            y_test,
            feature_names=data.get("feature_names"),
            target_names=data.get("target_names"),
            description=data.get("DESCR"),
        )


class GroupedDataset(Dataset):
    """Class that groups data-points.
    Used for calculating Shapley values of coalitions."""

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        data_groups: Sequence,
        feature_names: Optional[Sized] = None,
        target_names: Optional[Sized] = None,
        description: Optional[str] = None,
    ):
        """Class for grouping datasets.

        :param x_train: train input data
        :param y_train: labels of train data
        :param x_test: input of test data
        :param y_test: labels of test data
        :param data_groups: Iterable of the same length of x_train.
            For each train data-point, it associates a group label (which could be of any type, e.g. string or int).
            Data-points with the same label will then be grouped withing the GroupedDataset class.
        :param feature_names: name of the features of input data
        :param target_names: name of target data
        :param description: description of the dataset
        """
        super().__init__(
            x_train, y_train, x_test, y_test, feature_names, target_names, description
        )
        if len(data_groups) != len(x_train):
            raise ValueError(
                f"data_groups and x_train must have the same length. Instead got {len(data_groups)=} and {len(x_train)=}"
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
        These are not the indices of all the dataset, but only those referencing the groups."""
        return np.array(self._indices)

    @property
    def data_names(self):
        """Name given to the groups."""
        return list(self.groups.keys())

    def get_train_data(self, train_indices):
        """Given a set of indices, it returns the related groups."""
        data_indices = [
            idx for group_id in train_indices for idx in self.group_items[group_id][1]
        ]
        return super().get_train_data(data_indices)

    @classmethod
    def from_sklearn(
        cls,
        data: Bunch,
        train_size: float = 0.8,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> "GroupedDataset":
        """Constructs a Dataset object from an sklearn bunch as returned
        by the load_* functions in `sklearn.datasets` and groups it.

        :param data: sklearn dataset
        :param train_size: size of the training dataset. Used in train_test_split
        :param random_state: seed for train test split
        :param data_groups: for each element in the training set, it associates a group
            index or name.
        :return: Dataset with the selected sklearn data
        """
        data_groups: Optional[List] = kwargs.get("data_groups")
        if data_groups is None:
            raise ValueError("data_groups argument is missing")
        dataset = super().from_sklearn(data, train_size, random_state)
        return cls.from_dataset(dataset, data_groups)

    @classmethod
    def from_dataset(
        cls, dataset: Dataset, data_groups: Sequence[Any]
    ) -> "GroupedDataset":
        """Given a dataset, it makes it into a grouped dataset by passing a list of data
        groups, one for each element in the training set.

        :param dataset: Dataset object
        :param data_groups: for each element in the training set, it associates a group
            index or name.
        :return: GroupedDataset, with the initial Dataset grouped by data_groups.
        """
        return GroupedDataset(
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
    More info on the dataset can be found
    at https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify.

    If this method is called within the CI pipeline, it will load a reduced version of the dataset
    for test purposes.
    :param val_size: size of the validation set
    :param test_size: size of the test set
    :param min_year: minimum year of the returned data
    :param target_column: column to be returned as y (labels)
    :param random_state: fixes sklearn random seed
    :return: Tuple with 3 elements, each being a list sith [input_data, related_labels]
    """
    file_dir_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(
        file_dir_path, "../../../data/top_hits_spotify_dataset.csv"
    )
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
    else:
        url = "https://github.com/appliedAI-Initiative/valuation/blob/notebook_and_shapley_interface/data/top_hits_spotify_dataset.csv"
        data = pd.read_csv(url)
        data.to_csv(file_path, index=False)

    data = data[data["year"] > min_year]
    # TODO reading off an env variable within the method is dirty. Look into other solutions
    # to switching to reduced dataset when testing
    CI = os.environ.get("CI") in ("True", "true")
    if CI:
        data = data.iloc[:3]

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


def load_wine_dataset(train_size, test_size, random_seed=None):
    """
    Loads the sklearn wine dataset. More info can be found at 
    https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset
    :param train_size: fraction of points used for training dataset
    :param test_size: fraction of points used for test dataset
    :param random_seed: fix random seed. If None, no random seed is set.
    :returns: A tuple of four elements with the first three being \
        input and target values in the form of matrices of shape [NxD] the first and [N] the second. \
        The fourth element is a list containing names of features of the model.
    """

    import torch

    wine_bunch = load_wine(as_frame=True)
    x, x_test, y, y_test = train_test_split(
        wine_bunch.data,
        wine_bunch.target,
        train_size=1 - test_size,
        random_state=random_seed,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, train_size=train_size / (1 - test_size), random_state=random_seed
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
    """
    Sample from a uniform Gaussian mixture model.
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

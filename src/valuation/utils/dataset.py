import os
from collections import OrderedDict
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from numpy.lib.index_tricks import IndexExpression
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch, check_X_y


class Dataset:
    """Class for better handling datasets in the Dval library"""

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Iterable = None,
        target_names: Iterable = None,
        data_names: Iterable = None,
        description: str = None,
    ):
        """Class for better handling datasets in the Dval library

        :param x_train: train input data
        :param y_train: labels of train data
        :param x_test: input of test data
        :param y_test: labels of test data
        :param feature_names: name of the features of input data
        :param target_names: name of target data
        :param data_names: optional name for each entry in the train dataset.
            Must have the same length as x_train.
            For example input data may be indexed by timestamp (as it is typical for timeseries).
            In order to refer to them with such timestamps, one can pass the times column to data_names.
        :param description: description of the dataset
        """
        self.x_train, self.y_train = check_X_y(x_train, y_train)
        self.x_test, self.y_test = check_X_y(x_test, y_test)

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

        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else make_names("x", x_train)
        )
        self.target_names = (
            list(target_names) if target_names is not None else make_names("y", y_train)
        )

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
        self._data_names = list(data_names) if data_names is not None else self._indices

    def feature(self, name: str) -> IndexExpression:
        try:
            return np.index_exp[:, self.feature_names.index(name)]
        except ValueError:
            raise ValueError(f"Feature {name} is not in {self.feature_names}")

    def get_train_data(self, train_indices: List[int]):
        """Given a set of indices, it returns the train data that refer to those indices.
        This is used when calling different sub-sets of indices to calculate data shapley values.
        Notice that train_indices is not typically equal to the full indices, but only a subset of it.
        """
        x = self.x_train[train_indices]
        y = self.y_train[train_indices]
        return x, y

    def target(self, name: str) -> IndexExpression:
        try:
            return np.index_exp[:, self.target_names.index(name)]
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
        cls, data: Bunch, train_size: float = 0.8, random_state: int = None
    ) -> "Dataset":
        """Constructs a Dataset object from an sklearn bunch as returned
        by the load_* functions in `sklearn.datasets`
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
            data_names=data.get("data_names"),
            description=data.get("DESCR"),
        )

    try:
        import pandas as pd

        @classmethod
        def from_pandas(cls, df: pd.DataFrame) -> "Dataset":
            """That."""
            raise NotImplementedError

    except ModuleNotFoundError:
        pass


class GroupedDataset(Dataset):
    """Class that groups data-points.
    Useful for calculating Shapley values of coalitions."""

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        data_groups: Iterable,
        feature_names: Optional[Iterable] = None,
        target_names: Optional[Iterable] = None,
        description: Optional[str] = None,
    ):
        """Class for better grouped datasets.

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

        self.groups = OrderedDict({k: [] for k in set(data_groups)})
        for idx, group in enumerate(data_groups):
            self.groups[group].append(idx)
        self.group_items = list(self.groups.items())
        self._indices = list(range(len(self.groups.keys())))

    def __len__(self):
        return len(self.groups)

    @property
    def indices(self):
        """Indices of the grouped data points"""
        return np.array(self._indices)

    @property
    def data_names(self):
        return list(self.groups.keys())

    def get_train_data(self, train_indices):
        data_indices = [
            idx for group_id in train_indices for idx in self.group_items[group_id][1]
        ]
        return super().get_train_data(data_indices)

    @classmethod
    def from_sklearn(
        cls,
        data: Bunch,
        data_groups: List,
        train_size: float = 0.8,
        random_state: int = None,
    ) -> "GroupedDataset":
        dataset = super().from_sklearn(data, train_size, random_state)
        return cls.from_dataset(dataset, data_groups)

    @classmethod
    def from_dataset(cls, dataset: Dataset, data_groups: List):
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


def polynomial(coefficients, x):
    powers = np.arange(len(coefficients))
    return np.power(x, np.tile(powers, (len(x), 1)).T).T @ coefficients


def polynomial_dataset(coefficients: np.ndarray):
    """Coefficients must be for monomials of increasing degree"""
    from sklearn.utils import Bunch

    x = np.arange(-1, 1, 0.2)
    locs = polynomial(coefficients, x)
    y = np.random.normal(loc=locs, scale=0.3)
    db = Bunch()
    db.data, db.target = x.reshape(-1, 1), y
    poly = [f"{c} x^{i}" for i, c in enumerate(coefficients)]
    poly = " + ".join(poly)
    db.DESCR = f"$y \\sim N({poly}, 1)$"
    db.feature_names = ["x"]
    db.target_names = ["y"]
    return Dataset.from_sklearn(data=db, train_size=0.5), coefficients


def load_spotify_dataset(
    val_size: float,
    test_size: float,
    min_year: int = 2014,
    target_column: str = "popularity",
    random_state: int = 42,
):
    """Load spotify music dataset and selects song after min_year.
    If os. is True, it returns a small dataset for testing purposes."""
    CI = os.environ.get("CI") in ("True", "true")
    data = pd.read_csv("../data/top_hits_spotify_dataset.csv")
    data["genre"] = data["genre"].astype("category").cat.codes
    if CI:
        data = data.iloc[:3]
    else:
        data = data[data["year"] > min_year]
    y = data[target_column]
    X = data.drop(target_column, axis=1)
    X, X_test, y, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state
    )
    return [X_train, y_train], [X_val, y_val], [X_test, y_test]

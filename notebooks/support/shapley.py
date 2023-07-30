from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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
    root_dir_path = Path(__file__).parent.parent.parent
    file_path = root_dir_path / "data/top_hits_spotify_dataset.csv"
    if file_path.exists():
        data = pd.read_csv(file_path)
    else:
        url = "https://raw.githubusercontent.com/appliedAI-Initiative/pyDVL/develop/data/top_hits_spotify_dataset.csv"
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
        x, y, train_size=1 - test_size, random_state=random_seed
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

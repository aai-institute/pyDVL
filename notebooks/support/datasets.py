from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, overload

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.datasets import load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, TargetEncoder

from notebooks.support.common import filecache
from pydvl.utils import try_torch_import
from pydvl.valuation.dataset import Dataset

__all__ = ["load_digits_dataset", "load_spotify_dataset", "load_wine_dataset"]

torch = try_torch_import()

# Define Tensor type properly for type checking
if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any if torch is None else torch.Tensor

logger = logging.getLogger(__name__)


@overload
def load_digits_dataset(
    train_size: float,
    random_state: Optional[int] = None,
    use_tensors: Literal[False] = False,
    device: str | None = "cpu",
    shared_mem: bool = False,
    half_precision: bool = False,
) -> tuple[Dataset[NDArray], Dataset[NDArray]]: ...


@overload
def load_digits_dataset(
    train_size: float,
    random_state: Optional[int] = None,
    use_tensors: Literal[True] = True,
    device: str | None = "cpu",
    shared_mem: bool = False,
    half_precision: bool = False,
) -> tuple[Dataset[Tensor], Dataset[Tensor]]: ...


def load_digits_dataset(
    train_size: float,
    random_state: Optional[int] = None,
    device: str | None = None,
    shared_mem: bool = False,
    half_precision: bool = False,
) -> tuple[Dataset, Dataset]:
    """Loads the sklearn handwritten digits dataset.

    More info can be found at
    https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset.

    Args:
        train_size: Fraction of points used for training.
        random_state: Fix random seed. If `None`, the default rng is taken.
        device: If `None`, the data is stored as numpy arrays. Otherwise, torch.Tensors
            are used and moved to the selected device
        shared_mem: If `True`, the tensors are moved to shared memory.
    Returns
        A tuple of tra ining and test Datasets
    """

    bunch = load_digits(as_frame=True)
    if device is not None:
        dtype: Type = torch.float16 if half_precision else torch.float32
        x = torch.tensor(bunch.data.values / 16.0, dtype=dtype, device=device).reshape(
            -1, 1, 8, 8
        )
        y = torch.tensor(bunch.target.values, dtype=torch.long, device=device)
        mb = (x.nbytes + y.nbytes) / 2**20
        if shared_mem:
            x.share_memory_()
            y.share_memory_()
            logger.debug(
                f"Loaded {len(x)} data points to {x.device} ({mb:.2f}MB, {shared_mem=})"
            )
    else:
        dtype = np.float16 if half_precision else np.float32
        x = np.array(bunch.data.values / 16.0, dtype=dtype)
        y = np.array(bunch.target.values, dtype=np.int32)
        mb = (x.nbytes + y.nbytes) / 2**20
        logger.debug(f"Loaded {len(x)} data points ({mb:.2f}MB, mmap={shared_mem})")

    train, test = Dataset.from_arrays(
        x,
        y,
        train_size=train_size,
        random_state=random_state,
        stratify_by_target=True,
        mmap=device is None and shared_mem,
        target_names=["label"],
    )
    # TODO: Is this helping?
    del x, y
    import gc

    gc.collect()
    return train, test


def process_imgnet_io(
    df: pd.DataFrame, labels: dict, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = df["normalized_images"]
    y = df["labels"]
    ds_label_to_model_label = {
        ds_label: idx for idx, ds_label in enumerate(labels.values())
    }
    x_nn = torch.stack(x.tolist()).to(device)
    y_nn = torch.tensor([ds_label_to_model_label[yi] for yi in y], device=device)
    return x_nn, y_nn


def load_spotify_dataset(
    val_size: float,
    test_size: float,
    min_year: int = 2014,
    target_column: str = "popularity",
    random_state: int = 24,
) -> Tuple[List[NDArray], List[NDArray], List[NDArray]]:
    """Loads (and downloads if not already cached) the spotify music dataset.
    More info on the dataset can be found at
    https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify.

    If this method is called within the CI pipeline, it will load a reduced
    version of the dataset for testing purposes.

    Args:
        val_size: size of the validation set
        test_size: size of the test set
        min_year: minimum year of the returned data
        target_column: column to be returned as y (labels)
        random_state: fixes sklearn random seed

    Returns:
        Tuple with 3 elements, each of one of them is a list with [input_data, target_labels]
    """
    root_dir_path = Path(__file__).parent.parent.parent
    file_path = root_dir_path / "data/top_hits_spotify_dataset.csv"
    if file_path.exists():
        data = pd.read_csv(file_path)
    else:
        url = "https://raw.githubusercontent.com/aai-institute/pyDVL/develop/data/top_hits_spotify_dataset.csv"
        data = pd.read_csv(url)
        data.to_csv(file_path, index=False)

    data = data[data["year"] > min_year]
    data["genre"] = data["genre"].astype("category").cat.codes
    y = data[target_column]
    X = data.drop(target_column, axis=1)

    # Drop the 'song' column
    X = X.drop(columns=["song"])

    # Train, val, test splits
    X, X_test, y, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state
    )

    return (
        [X_train, y_train],
        [X_val, y_val],
        [X_test, y_test],
    )


def load_wine_dataset(
    train_size: float, test_size: float, random_state: Optional[int] = None
) -> Tuple[
    Tuple[NDArray, NDArray], Tuple[NDArray, NDArray], Tuple[NDArray, NDArray], List[str]
]:
    """Loads the sklearn wine dataset. More info can be found at
    https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset.

    Args:
        train_size: fraction of points used for training dataset
        test_size: fraction of points used for test dataset
        random_state: fix random seed. If None, no random seed is set.
    Returns:
        A tuple of four elements with the first three being input and
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
) -> Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]]:
    """Sample from a uniform Gaussian mixture model.

    Args:
        mus: 2d-matrix [CxD] with the means of the components in the rows.
        sigma: Standard deviation of each dimension of each component.
        num_samples: The number of samples to generate.
        train_size: fraction of points used for training dataset
        test_size: fraction of points used for test dataset
        random_seed: fix random seed. If None, no random seed is set.

    Returns:
        A tuple of matrix x of shape [NxD] and target vector y of shape [N].
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


DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_preprocess_imagenet(
    train_size: float,
    test_size: float,
    downsampling_ratio: float = 1,
    keep_labels: Optional[dict] = None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the tiny imagenet dataset from huggingface and preprocesses it
    for model input.

    Args:
        train_size: fraction of indices to use for training
        test_size: fraction of data to use for testing
        downsampling_ratio: which fraction of the full dataset to keep.
            E.g. downsample_ds_to_fraction=0.2 only 20% of the dataset is kept
        keep_labels: which of the original labels to keep and their names.
            E.g. keep_labels={10:"a", 20: "b"} only returns the images with labels
            10 and 20 and changes the values to "a" and "b" respectively.
        random_state: Random state. Fix this for reproducibility of sampling.

    Returns:
        a tuple of three dataframes, first holding the training data,
        second validation, third test.

        Each has 3 keys: normalized_images has all the input images, rescaled
        to mean 0.5 and std 0.225,
        labels has the labels of each image, while images has the unmodified
        PIL images.
    """
    try:
        from datasets import load_dataset, utils
        from torchvision import transforms
    except ImportError as e:
        raise RuntimeError(
            "Torchvision and Huggingface datasets are required to load and "
            "process the imagenet dataset."
        ) from e

    utils.logging.set_verbosity_error()

    preprocess_rgb = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225]),
        ]
    )

    def _process_dataset(ds):
        processed_ds = {"normalized_images": [], "labels": [], "images": []}
        for i, item in enumerate(ds):
            if item["image"].mode == "RGB":
                processed_ds["normalized_images"].append(preprocess_rgb(item["image"]))
                processed_ds["images"].append(item["image"])
                if keep_labels is not None:
                    processed_ds["labels"].append(keep_labels[item["label"]])
                else:
                    processed_ds["labels"].append(item["label"])
        return pd.DataFrame.from_dict(processed_ds)

    def split_ds_by_size(dataset, split_size):
        split_ds = dataset.train_test_split(
            train_size=split_size,
            seed=random_state,
            stratify_by_column="label",
        )
        return split_ds

    dataset_path = "Maysee/tiny-imagenet"

    if os.environ.get("CI"):
        tiny_imagenet = load_dataset(dataset_path, split="valid")
        if keep_labels is not None:
            tiny_imagenet = tiny_imagenet.filter(
                lambda item: item["label"] in keep_labels
            )
        split = tiny_imagenet.shard(2, 0)
        tiny_imagenet_test = tiny_imagenet.shard(2, 1)
        tiny_imagenet_train = split.shard(5, 0)
        tiny_imagenet_val = split.shard(5, 1)
        train_ds = _process_dataset(tiny_imagenet_train)
        val_ds = _process_dataset(tiny_imagenet_val)
        test_ds = _process_dataset(tiny_imagenet_test)
        return train_ds, val_ds, test_ds

    tiny_imagenet = load_dataset(dataset_path, split="train")

    if downsampling_ratio != 1:
        tiny_imagenet = tiny_imagenet.shard(
            num_shards=int(1 / downsampling_ratio), index=0
        )
    if keep_labels is not None:
        tiny_imagenet = tiny_imagenet.filter(
            lambda item: item["label"] in keep_labels.keys()
        )

    split_ds = split_ds_by_size(tiny_imagenet, 1 - test_size)
    test_ds = _process_dataset(split_ds["test"])

    split_ds = split_ds_by_size(split_ds["train"], train_size)
    train_ds = _process_dataset(split_ds["train"])
    val_ds = _process_dataset(split_ds["test"])

    return train_ds, val_ds, test_ds


def corrupt_imagenet(
    dataset: pd.DataFrame,
    fraction_to_corrupt: float,
    avg_influences: NDArray[np.float64],
) -> Tuple[pd.DataFrame, Dict[Any, List[int]]]:
    """Given the preprocessed tiny imagenet dataset (or a subset of it),
    it takes a fraction of the images with the highest influence and (randomly)
    flips their labels.

    Args:
        dataset: preprocessed tiny imagenet dataset
        fraction_to_corrupt: float, fraction of data to corrupt
        avg_influences: average influences of each training point on the test set in the \
            non-corrupted case.
    Returns:
        first element is the corrupted dataset, second is the list of indices \
        related to the images that have been corrupted.
    """
    labels = dataset["labels"].unique()
    corrupted_dataset = deepcopy(dataset)
    corrupted_indices = {l: [] for l in labels}

    avg_influences_series = pd.DataFrame()
    avg_influences_series["avg_influences"] = avg_influences
    avg_influences_series["labels"] = dataset["labels"]

    for label in labels:
        class_data = avg_influences_series[avg_influences_series["labels"] == label]
        num_corrupt = int(fraction_to_corrupt * len(class_data))
        indices_to_corrupt = class_data.nlargest(
            num_corrupt, "avg_influences"
        ).index.tolist()
        wrong_labels = [l for l in labels if l != label]
        for img_idx in indices_to_corrupt:
            sample_label = np.random.choice(wrong_labels)
            corrupted_dataset.at[img_idx, "labels"] = sample_label
            corrupted_indices[sample_label].append(img_idx)
    return corrupted_dataset, corrupted_indices


@filecache(path=DATA_DIR / "adult_data_raw.pkl")
def load_adult_data_raw() -> pd.DataFrame:
    """
    Downloads the adult dataset from UCI and returns it as a pandas DataFrame.

    Returns:
        The adult dataset as a pandas DataFrame.
    """
    data_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    )

    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    data_types = {
        "age": int,
        "workclass": "category",
        "fnlwgt": int,
        "education": "category",
        "education-num": int,  # increasing level of education
        "marital-status": "category",
        "occupation": "category",
        "relationship": "category",
        "race": "category",
        "sex": "category",
        "capital-gain": int,
        "capital-loss": int,
        "hours-per-week": int,
        "native-country": "category",
        "income": "category",
    }

    return pd.read_csv(
        data_url,
        names=column_names,
        sep=r",\s*",
        engine="python",
        na_values="?",
        dtype=data_types,
    )


def load_adult_data(
    train_size: float = 0.7,
    subsample: float = 1.0,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[Dataset, Dataset]:
    """
    Loads the adult dataset from UCI and performs some preprocessing.

    The data is preprocessed by performing target encoding of the categorical variables,
    dropping the "education" column and dropping NaNs

    Ideally the encoding would be done in a pipeline, but we are trying to remove as
    much complexity from the notebooks as possible.

    Args:
        subsample: fraction of the data to keep. Range [0,1]
        train_size: fraction of the (subsampled) data to use for training
        random_state: random state for reproducibility

    Returns:
        A tuple with training and test datasets.
    """

    df = load_adult_data_raw(**kwargs)
    if subsample < 1:
        df = df.sample(frac=subsample, random_state=random_state)
    column_names = df.columns.tolist()

    df["income"] = df["income"].cat.codes
    df.drop(columns=["education"], inplace=True)  # education-num is enough
    df.dropna(inplace=True)
    column_names.remove("education")
    column_names.remove("income")

    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(columns=["income"]).values,
        df["income"].values,
        train_size=train_size,
        random_state=random_state,
        stratify=df["income"].values,
    )

    te = TargetEncoder(target_type="binary", random_state=random_state)
    x_train = te.fit_transform(x_train, y_train)
    x_test = te.transform(x_test)

    return (
        Dataset(x_train, y_train, feature_names=column_names, target_names=["income"]),
        Dataset(x_test, y_test, feature_names=column_names, target_names=["income"]),
    )


def to_dataframe(dataset: Dataset) -> pd.DataFrame:
    """
    Converts a dataset to a pandas DataFrame

    Args:
        dataset: Dataset to convert

    Returns:
        A pandas DataFrame
    """
    y = dataset.y[:, np.newaxis] if dataset.y.ndim == 1 else dataset.y
    df = pd.DataFrame(dataset.x, columns=dataset.feature_names).assign(
        **{name: y[:, i] for i, name in enumerate(dataset.target_names)}
    )
    return df

import logging
import os
from collections import namedtuple
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL.JpegImagePlugin import JpegImageFile

from pydvl.utils import Dataset

logger = logging.getLogger(__name__)
Losses = namedtuple("Losses", "training val")


def plot_gaussian_blobs(
    train_ds: Tuple[NDArray[np.float_], NDArray[np.int_]],
    test_ds: Tuple[NDArray[np.float_], NDArray[np.int_]],
    x_min: Optional[NDArray[np.float_]] = None,
    x_max: Optional[NDArray[np.float_]] = None,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_title: Optional[str] = None,
    vline: Optional[float] = None,
    line: Optional[NDArray[np.float_]] = None,
    suptitle: Optional[str] = None,
    s: Optional[float] = None,
    figsize: Tuple[int, int] = (20, 10),
):
    """Plots training and test data in two separate plots, with the optimal
    decision boundary as passed to the line argument.

    :param train_ds: A 2-element tuple with training data and labels thereof.
        Features have shape `(N, 2)` and the target_variable has shape `(n,)`.
    :param test_ds: A 2-element tuple with test data and labels. Same format as
        train_ds.
    :param x_min: Set to define the minimum boundaries of the plot.
    :param x_max: Set to define the maximum boundaries of the plot.
    :param line: Optional, line of shape (M,2), where each row is a point of the
        2-d line.
    :param s: The thickness of the points to plot.
    :param figsize: for `plt.figure()`
    """

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    spec = fig.add_gridspec(20, 2)
    ax = [fig.add_subplot(spec[:-1, i]) for i in range(2)]
    ax.append(fig.add_subplot(spec[-1, :]))

    datasets = {"train": train_ds, "test": test_ds}

    discrete_keys = [
        key for key, dataset in datasets.items() if dataset[1].dtype == int
    ]
    if 0 < len(discrete_keys) < len(datasets):
        "You can only plot either discrete or only continuous plots."

    cmap = plt.get_cmap("Set1")
    all_y = np.concatenate(tuple([v[1] for _, v in datasets.items()]), axis=0)
    unique_y = np.sort(np.unique(all_y))
    num_classes = len(unique_y)
    handles = [mpatches.Patch(color=cmap(i), label=y) for i, y in enumerate(unique_y)]

    for i, dataset_name in enumerate(datasets.keys()):
        x, y = datasets[dataset_name]
        if x.shape[1] != 2:
            raise AttributeError("The maximum number of allowed features is 2.")

        ax[i].set_title(dataset_name)
        if x_min is not None:
            ax[i].set_xlim(x_min[0], x_max[0])  # type: ignore
        if x_max is not None:
            ax[i].set_ylim(x_min[1], x_max[1])  # type: ignore

        if line is not None:
            ax[i].plot(line[:, 0], line[:, 1], color="black")

        ax[i].scatter(x[:, 0], x[:, 1], c=cmap(y), s=s, edgecolors="black")

        if xlabel is not None:
            ax[i].set_xlabel(xlabel)
        if ylabel is not None:
            ax[i].set_ylabel(ylabel)

        if vline is not None:
            ax[i].axvline(vline, color="black", linestyle="--")

    ax[-1].legend(handles=handles, loc="center", ncol=num_classes, title=legend_title)
    ax[-1].axis("off")

    if suptitle is not None:
        plt.suptitle(suptitle)

    plt.show()


def plot_influences(
    x: NDArray[np.float_],
    influences: NDArray[np.float_],
    corrupted_indices: Optional[List[int]] = None,
    *,
    ax: Optional[plt.Axes] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_title: Optional[str] = None,
    line: Optional[NDArray[np.float_]] = None,
    suptitle: Optional[str] = None,
    colorbar_limits: Optional[Tuple] = None,
) -> plt.Axes:
    """Plots the influence values of the training data with a color map.

    :param x: Input to the model, of shape (N,2) with N being the total number
        of points.
    :param influences: an array  of shape (N,) with influence values for each
        data point.
    :param line: Optional, line of shape [Mx2], where each row is a point of the
        2-dimensional line. (??)
    """
    if ax is None:
        _, ax = plt.subplots()
    sc = ax.scatter(x[:, 0], x[:, 1], c=influences)
    if line is not None:
        ax.plot(line[:, 0], line[:, 1], color="black")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if suptitle is not None:
        plt.suptitle(suptitle)

    plt.colorbar(sc, label=legend_title)
    if colorbar_limits is not None:
        sc.clim(*colorbar_limits)
    if corrupted_indices is not None:
        ax.scatter(
            x[:, 0][corrupted_indices],
            x[:, 1][corrupted_indices],
            facecolors="none",
            edgecolors="r",
            s=80,
        )
    return ax


def plot_iris(
    data: Dataset,
    indices: List[int] = None,
    highlight_indices: Optional[Sequence[int]] = None,
    suptitle: str = None,
    legend_title: str = None,
    legend_labels: Sequence[str] = None,
    colors: Iterable = None,
    colorbar_limits: Optional[Tuple] = None,
    figsize: Tuple[int, int] = (20, 8),
):
    """Scatter plots for the iris dataset.

    :param data: a Dataset with a valid train / test split
    :param indices: subset of `data.indices`.
    :param highlight_indices: circle these indices in red
    :param suptitle: centered title for the figure
    :param legend_title: A title for the legend
    :param legend_labels: Labels for the legend entries
    :param colors: use with indices to set the color (e.g. to values).
    :param colorbar_limits: Range of values to display in the colorbar. A
        colorbar will only be displayed if there are more than 10 colors.
    :param figsize: Size of figure for matplotlib
    """
    if indices is not None:
        x_train = data.x_train[indices]
        y_train = data.y_train[indices]
    else:
        x_train = data.x_train
        y_train = data.y_train

    sepal_length_indices = data.feature("sepal length (cm)")
    sepal_width_indices = data.feature("sepal width (cm)")
    petal_length_indices = data.feature("petal length (cm)")
    petal_width_indices = data.feature("petal width (cm)")

    if colors is None:
        colors = y_train

    def _handle_legend(scatter):
        if len(np.unique(colors)) > 10:
            plt.colorbar(label=legend_title)
            if colorbar_limits is not None:
                plt.clim(*colorbar_limits)
        else:
            plt.legend(
                handles=scatter.legend_elements()[0],
                labels=legend_labels,
                title=legend_title,
            )

    plt.figure(figsize=figsize)
    plt.suptitle(suptitle)
    plt.subplot(1, 2, 1)
    xmin, xmax = (
        x_train[sepal_length_indices].min(),
        x_train[sepal_length_indices].max(),
    )
    ymin, ymax = (
        x_train[sepal_width_indices].min(),
        x_train[sepal_width_indices].max(),
    )
    xmargin = 0.1 * (xmax - xmin)
    ymargin = 0.1 * (ymax - ymin)
    scatter = plt.scatter(
        x_train[sepal_length_indices],
        x_train[sepal_width_indices],
        c=colors,
        marker="o",
        alpha=0.8,
    )
    plt.xlim(xmin - xmargin, xmax + xmargin)
    plt.ylim(ymin - ymargin, ymax + ymargin)
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    _handle_legend(scatter)
    if highlight_indices is not None:
        scatter = plt.scatter(
            x_train[sepal_length_indices][highlight_indices],
            x_train[sepal_width_indices][highlight_indices],
            facecolors="none",
            edgecolors="r",
            s=80,
        )

    plt.subplot(1, 2, 2)
    xmin, xmax = (
        x_train[petal_length_indices].min(),
        x_train[petal_length_indices].max(),
    )
    ymin, ymax = (
        x_train[petal_width_indices].min(),
        x_train[petal_width_indices].max(),
    )
    xmargin = 0.1 * (xmax - xmin)
    ymargin = 0.1 * (ymax - ymin)
    scatter = plt.scatter(
        x_train[petal_length_indices],
        x_train[petal_width_indices],
        c=colors,
        marker="o",
        alpha=0.8,
    )
    plt.xlim(xmin - xmargin, xmax + xmargin)
    plt.ylim(ymin - ymargin, ymax + ymargin)
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")
    _handle_legend(scatter)
    if highlight_indices is not None:
        scatter = plt.scatter(
            x_train[petal_length_indices][highlight_indices],
            x_train[petal_width_indices][highlight_indices],
            facecolors="none",
            edgecolors="r",
            s=80,
        )


def load_preprocess_imagenet(
    train_size: float,
    test_size: float,
    downsampling_ratio: float = 1,
    keep_labels: Optional[dict] = None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the tiny imagenet dataset from huggingface and preprocesses it
    for model input.

    :param train_size: fraction of indices to use for training
    :param test_size: fraction of data to use for testing
    :param downsampling_ratio: which fraction of the full dataset to keep.
        E.g. downsample_ds_to_fraction=0.2 only 20% of the dataset is kept
    :param keep_labels: which of the original labels to keep and their names.
        E.g. keep_labels={10:"a", 20: "b"} only returns the images with labels
        10 and 20 and changes the values to "a" and "b" respectively.
    :param random_state: Random state. Fix this for reproducibility of sampling.
    :return: a tuple of three dataframes, first holding the training data,
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

    if os.environ.get("CI"):
        tiny_imagenet = load_dataset("Maysee/tiny-imagenet", split="valid")
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

    tiny_imagenet = load_dataset("Maysee/tiny-imagenet", split="train")

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


def plot_sample_images(dataset: pd.DataFrame, n_images_per_class: int = 3):
    """Plots several images for each class of a pre-processed imagenet dataset
    (or a subset of it).

    :param dataset: imagenet dataset
    :param n_images_per_class: number of images per class to plot
    """
    labels = dataset["labels"].unique()
    fig, axes = plt.subplots(nrows=n_images_per_class, ncols=len(labels))
    fig.suptitle("Examples of training images")
    for class_idx, class_label in enumerate(labels):
        for img_idx, (_, img_data) in enumerate(
            dataset[dataset["labels"] == class_label].iterrows()
        ):
            axes[img_idx, class_idx].imshow(img_data["images"])
            axes[img_idx, class_idx].axis("off")
            axes[img_idx, class_idx].set_title(f"img label: {class_label}")
            if img_idx + 1 >= n_images_per_class:
                break
    plt.show()


def plot_lowest_highest_influence_images(
    subset_influences: NDArray[np.float_],
    subset_images: List[JpegImageFile],
    num_to_plot: int,
):
    """Given a set of images and their influences over another, plots two columns
    of `num_to_plot` images each. Those on the right column have the lowest influence,
     those on the right the highest.

    :param subset_influences: an array with influence values
    :param subset_images: a list of images
    :param num_to_plot: int, number of high and low influence images to plot
    """
    top_if_idxs = np.argsort(subset_influences)[-num_to_plot:]
    bottom_if_idxs = np.argsort(subset_influences)[:num_to_plot]

    fig, axes = plt.subplots(nrows=num_to_plot, ncols=2)
    fig.suptitle("Lowest (left) and highest (right) influences")

    for plt_idx, img_idx in enumerate(bottom_if_idxs):
        axes[plt_idx, 0].set_title(f"img influence: {subset_influences[img_idx]:0f}")
        axes[plt_idx, 0].imshow(subset_images[img_idx])
        axes[plt_idx, 0].axis("off")

    for plt_idx, img_idx in enumerate(top_if_idxs):
        axes[plt_idx, 1].set_title(f"img influence: {subset_influences[img_idx]:0f}")
        axes[plt_idx, 1].imshow(subset_images[img_idx])
        axes[plt_idx, 1].axis("off")

    plt.show()


def plot_losses(losses: Losses):
    """Plots the train and validation loss

    :param training_loss: list of training losses, one per epoch
    :param validation_loss: list of validation losses, one per epoch
    """
    _, ax = plt.subplots()
    ax.plot(losses.training, label="Train")
    ax.plot(losses.val, label="Val")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Train epoch")
    ax.legend()
    plt.show()


def corrupt_imagenet(
    dataset: pd.DataFrame,
    fraction_to_corrupt: float,
    avg_influences: NDArray[np.float_],
) -> Tuple[pd.DataFrame, Dict[Any, List[int]]]:
    """Given the preprocessed tiny imagenet dataset (or a subset of it), 
    it takes a fraction of the images with the highest influence and (randomly)
    flips their labels.

    :param dataset: preprocessed tiny imagenet dataset
    :param fraction_to_corrupt: float, fraction of data to corrupt
    :param avg_influences: average influences of each training point on the test set in the \
        non-corrupted case.
    :return: first element is the corrupted dataset, second is the list of indices \
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


def compute_mean_corrupted_influences(
    corrupted_dataset: pd.DataFrame,
    corrupted_indices: Dict[Any, List[int]],
    avg_corrupted_influences: NDArray[np.float_],
) -> pd.DataFrame:
    """Given a corrupted dataset, it returns a dataframe with average influence for each class,
    separating corrupted and original points.

    :param corrupted_dataset: corrupted dataset as returned by get_corrupted_imagenet
    :param corrupted_indices: list of corrupted indices, as returned by get_corrupted_imagenet
    :param avg_corrupted_influences: average influence of each training point on the test dataset
    :return: a dataframe holding the average influence of corrupted and non-corrupted data
    """
    labels = corrupted_dataset["labels"].unique()
    avg_label_influence = pd.DataFrame(
        columns=["label", "avg_non_corrupted_infl", "avg_corrupted_infl", "score_diff"]
    )
    for idx, label in enumerate(labels):
        avg_influences_series = pd.Series(avg_corrupted_influences)
        class_influences = avg_influences_series[corrupted_dataset["labels"] == label]
        corrupted_infl = class_influences[
            class_influences.index.isin(corrupted_indices[label])
        ]
        non_corrupted_infl = class_influences[
            ~class_influences.index.isin(corrupted_indices[label])
        ]
        avg_non_corrupted = np.mean(non_corrupted_infl)
        avg_corrupted = np.mean(corrupted_infl)
        avg_label_influence.loc[idx] = [
            label,
            avg_non_corrupted,
            avg_corrupted,
            avg_non_corrupted - avg_corrupted,
        ]
    return avg_label_influence


def plot_corrupted_influences_distribution(
    corrupted_dataset: pd.DataFrame,
    corrupted_indices: Dict[Any, List[int]],
    avg_corrupted_influences: NDArray[np.float_],
    figsize: Tuple[int, int] = (16, 8),
):
    """Given a corrupted dataset, plots the histogram with the distribution of
    influence values. This is done separately for each label: each has a plot
    where the distribution of the influence of non-corrupted points is compared
    to that of corrupted ones.

    :param corrupted_dataset: corrupted dataset as returned by
        get_corrupted_imagenet
    :param corrupted_indices: list of corrupted indices, as returned by
        get_corrupted_imagenet
    :param avg_corrupted_influences: average influence of each training point on
        the test dataset
    :param figsize: for `plt.subplots()`
    :return: a dataframe holding the average influence of corrupted and
        non-corrupted data
    """
    labels = corrupted_dataset["labels"].unique()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle("Distribution of corrupted and clean influences.")
    for idx, label in enumerate(labels):
        avg_influences_series = pd.Series(avg_corrupted_influences)
        class_influences = avg_influences_series[corrupted_dataset["labels"] == label]
        corrupted_infl = class_influences[
            class_influences.index.isin(corrupted_indices[label])
        ]
        non_corrupted_infl = class_influences[
            ~class_influences.index.isin(corrupted_indices[label])
        ]
        axes[idx].hist(non_corrupted_infl, label="Non corrupted", alpha=0.7)
        axes[idx].hist(corrupted_infl, label="Corrupted", alpha=0.7, color="green")
        axes[idx].set_xlabel("Influence values")
        axes[idx].set_ylabel("Number of samples")
        axes[idx].set_title(f"Influences for {label=}")
        axes[idx].legend()
    plt.show()

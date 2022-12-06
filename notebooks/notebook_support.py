from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cloudpickle import pickle as pkl

from pydvl.utils import Dataset

try:
    import torch

    _TORCH_INSTALLED = True
except ImportError:
    _TORCH_INSTALLED = False

if TYPE_CHECKING:
    from numpy.typing import NDArray

imgnet_model_data_path = Path().resolve().parent / "data/imgnet_model"


def plot_dataset(
    train_ds: Tuple["NDArray", "NDArray"],
    test_ds: Tuple["NDArray", "NDArray"],
    x_min: Optional["NDArray"] = None,
    x_max: Optional["NDArray"] = None,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_title: Optional[str] = None,
    vline: Optional[float] = None,
    line: Optional["NDArray"] = None,
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
    """

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    spec = fig.add_gridspec(20, 2)
    ax = [fig.add_subplot(spec[:-1, i]) for i in range(2)]
    ax.append(fig.add_subplot(spec[-1, :]))

    datasets = {
        "train": train_ds,
        "test": test_ds,
    }

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
    x: "NDArray",
    influences: "NDArray",
    corrupted_indices: Optional[List[int]] = None,
    *,
    ax: Optional[plt.Axes] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_title: Optional[str] = None,
    line: Optional["NDArray"] = None,
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
        plt.clim(*colorbar_limits)
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
    downsample_ds_to_fraction: float = 1,
    keep_labels: Optional[List] = None,
    random_state: Optional[int] = None,
    is_CI: bool = False,
):
    try:
        from datasets import load_dataset
        from torchvision import transforms
    except ImportError as e:
        raise RuntimeError(
            "Torchvision and Huggingface datasets are required to load and "
            "process the imagenet dataset."
        ) from e

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
                processed_ds["labels"].append(item["label"])
        return pd.DataFrame.from_dict(processed_ds)

    tiny_imagenet = load_dataset("Maysee/tiny-imagenet", split="train")
    if downsample_ds_to_fraction != 1:
        tiny_imagenet = tiny_imagenet.shard(1 / downsample_ds_to_fraction, 0)
    if keep_labels is not None:
        tiny_imagenet = tiny_imagenet.filter(lambda item: item["label"] in keep_labels)

    split_ds = tiny_imagenet.train_test_split(
        train_size=1 - test_size,
        seed=random_state,
    )
    test_ds = _process_dataset(split_ds["test"])

    split_ds = split_ds["train"].train_test_split(
        train_size=train_size,
        seed=random_state,
    )
    train_ds = _process_dataset(split_ds["train"])
    val_ds = _process_dataset(split_ds["test"])
    return train_ds, val_ds, test_ds


def save_model(model, train_loss, val_loss, model_name):
    torch.save(model.state_dict(), imgnet_model_data_path / f"{model_name}_weights.pth")
    with open(
        imgnet_model_data_path / f"{model_name}_train_val_loss.pkl", "wb"
    ) as file:
        pkl.dump([train_loss, val_loss], file)


def load_model(model, model_name):
    model.load_state_dict(
        torch.load(imgnet_model_data_path / f"{model_name}_weights.pth")
    )
    with open(
        imgnet_model_data_path / f"{model_name}_train_val_loss.pkl", "rb"
    ) as file:
        train_loss, val_loss = pkl.load(file)
    return train_loss, val_loss


def save_results(results, file_name):
    with open(imgnet_model_data_path / f"{file_name}", "wb") as file:
        pkl.dump(results, file)


def load_results(file_name):
    with open(imgnet_model_data_path / f"{file_name}", "rb") as file:
        results = pkl.load(file)
    return results


def plot_sample_images(
    dataset,
    labels,
    n_images_per_class=3,
    figsize=(8, 8),
):
    plt.rcParams["figure.figsize"] = figsize
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


def plot_top_bottom_if_images(
    subset_influences,
    subset_images,
    num_to_plot,
    figsize=(8, 8),
):
    top_if_idxs = np.argsort(subset_influences)[-num_to_plot:]
    bottom_if_idxs = np.argsort(subset_influences)[:num_to_plot]

    fig, axes = plt.subplots(nrows=num_to_plot, ncols=2)
    plt.rcParams["figure.figsize"] = figsize
    fig.suptitle("Botton (left) and top (right) influences")

    for plt_idx, img_idx in enumerate(bottom_if_idxs):
        axes[plt_idx, 0].set_title(f"img influence: {subset_influences[img_idx]:0f}")
        axes[plt_idx, 0].imshow(subset_images[img_idx])
        axes[plt_idx, 0].axis("off")

    for plt_idx, img_idx in enumerate(top_if_idxs):
        axes[plt_idx, 1].set_title(f"img influence: {subset_influences[img_idx]:0f}")
        axes[plt_idx, 1].imshow(subset_images[img_idx])
        axes[plt_idx, 1].axis("off")

    plt.show()


def plot_train_val_loss(train_loss, val_loss):
    _, ax = plt.subplots()
    ax.plot(train_loss, label="Train")
    ax.plot(val_loss, label="Val")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Train epoch")
    ax.legend()
    plt.show()


def get_corrupted_imagenet(
    dataset, labels_to_keep, fraction_to_corrupt, avg_influences
):
    indices_to_corrupt = []
    corrupted_dataset = deepcopy(dataset)
    corrupted_indices = {l: [] for l in labels_to_keep}

    avg_influences_series = pd.DataFrame()
    avg_influences_series["avg_influences"] = avg_influences
    avg_influences_series["labels"] = dataset["labels"]

    for label in labels_to_keep:
        class_data = avg_influences_series[avg_influences_series["labels"] == label]
        num_corrupt = int(fraction_to_corrupt * len(class_data))
        indices_to_corrupt = class_data.nlargest(
            num_corrupt, "avg_influences"
        ).index.tolist()
        wrong_labels = [l for l in labels_to_keep if l != label]
        for img_idx in indices_to_corrupt:
            sample_label = np.random.choice(wrong_labels)
            corrupted_dataset.at[img_idx, "labels"] = sample_label
            corrupted_indices[sample_label].append(img_idx)
    return corrupted_dataset, corrupted_indices


def plot_influence_distribution(
    corrupted_dataset,
    labels_to_keep,
    corrupted_indices,
    avg_corrupted_influences,
    figsize=(16, 8),
):
    plt.rcParams["figure.figsize"] = figsize
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Distribution of corrupted and clean influences.")
    avg_label_influence = pd.DataFrame(
        columns=["label", "avg_non_corrupted_infl", "avg_corrupted_infl"]
    )
    for idx, label in enumerate(labels_to_keep):
        avg_influences_series = pd.Series(avg_corrupted_influences)
        class_influences = avg_influences_series[corrupted_dataset["labels"] == label]
        corrupted_infl = class_influences[
            class_influences.index.isin(corrupted_indices[label])
        ]
        non_corrupted_infl = class_influences[
            ~class_influences.index.isin(corrupted_indices[label])
        ]
        avg_label_influence.loc[idx] = [
            label,
            np.mean(non_corrupted_infl),
            np.mean(corrupted_infl),
        ]
        axes[idx].hist(
            non_corrupted_infl, label="non corrupted data", density=True, alpha=0.7
        )
        axes[idx].hist(corrupted_infl, label="corrupted data", density=True, alpha=0.7)
        axes[idx].set_xlabel("influence values")
        axes[idx].set_ylabel("Points distribution")
        axes[idx].set_title(f"influences for {label=}")
        axes[idx].legend()
    plt.show()
    return avg_label_influence.astype({"label": "int32"})

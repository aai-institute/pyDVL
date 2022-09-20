from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from valuation.utils.dataset import Dataset

if TYPE_CHECKING:
    try:
        from numpy.typing import NDArray
    except ImportError:
        from numpy import ndarray as NDArray


def plot_shapley(
    dval_df: pd.DataFrame,
    figsize: Tuple[int, int] = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
):
    """Plots the shapley values, as returned from shapley.get_shapley_values.

    :param dval_df: dataframe with the shapley values
    :param figsize: tuple with figure size
    :param title: string, title of the plot
    :param xlabel: string, x label of the plot
    :param ylabel: string, y label of the plot
    """
    fig = plt.figure(figsize=figsize)
    plt.errorbar(
        x=dval_df["data_key"],
        y=dval_df["shapley_dval"],
        yerr=dval_df["dval_std"],
        fmt="o",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    return fig


def plot_dataset(
    train_ds: Tuple["NDArray", "NDArray"],
    test_ds: Tuple["NDArray", "NDArray"],
    x_min: Optional["NDArray"] = None,
    x_max: Optional["NDArray"] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_title: Optional[str] = None,
    vline: Optional[float] = None,
    line: Optional["NDArray"] = None,
    suptitle: Optional[str] = None,
    s: Optional[float] = None,
):
    """
    Plots a train and test data in two separate plots, with also the optimal decision boundary as passed to the
    line argument.
    :param train_ds: A 2-elements tuple with train input and labels. Note that the features have size [Nx2] and
        the target_variable [N].
    :param test_ds:  A 2-elements tuple with test input and labels. Same format as train_ds.
    :param x_min: Set to define the minimum boundaries of the plot.
    :param x_max: Set to define the maximum boundaries of the plot.
    :param line: Optional, line of shape [Mx2], where each row is a point of the 2-dimensional line.
    :param s: The thickness of the points to plot.
    """

    fig = plt.figure(figsize=(6 * 2, 4), constrained_layout=True)
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

    num_classes = None
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
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_title: Optional[str] = None,
    line: Optional["NDArray"] = None,
    suptitle: Optional[str] = None,
    s: Optional[float] = None,
):
    """
    Plots the influence values of the train data with a color map.
    :param x_train: Input to the model. Note that the it must have size [Nx2], with N being the total
        number of points.
    :param train_influences: an array with influence values for each data point. Must have size N.
    :param line: Optional, line of shape [Mx2], where each row is a point of the 2-dimensional line.
    :param s: The thickness of the points to plot.
    """
    ax = plt.subplot()
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
    plt.show()


def plot_iris(
    data: Dataset,
    indices: List[int] = None,
    suptitle: str = None,
    colors: Iterable = None,
    colorbal_label: str = None,
    plot_test: bool = False,
):
    """Scatter plots for the iris dataset.
    :param data: split Dataset.
    :param indices: subset of data.indices
    :param colors: use with indices to set the color (e.g. to values)
    :param plot_test: plots the points from the test set too.
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

    plt.figure(figsize=(16, 6))
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
    plt.scatter(
        x_train[sepal_length_indices],
        x_train[sepal_width_indices],
        c=colors,
        marker="1",
    )
    if plot_test:
        plt.scatter(
            data.x_test[sepal_length_indices],
            data.x_test[sepal_width_indices],
            c=data.y_test,
            marker="o",
            alpha=0.4,
        )
    plt.xlim(xmin - xmargin, xmax + xmargin)
    plt.ylim(ymin - ymargin, ymax + ymargin)
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    if colors is not None:
        plt.colorbar(label=colorbal_label)

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
    plt.scatter(
        x_train[petal_length_indices],
        x_train[petal_width_indices],
        c=colors,
        marker="1",
    )
    if plot_test:
        plt.scatter(
            data.x_test[petal_length_indices],
            data.x_test[petal_width_indices],
            c=data.y_test,
            marker="o",
            alpha=0.4,
        )
    plt.xlim(xmin - xmargin, xmax + xmargin)
    plt.ylim(ymin - ymargin, ymax + ymargin)
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")
    if colors is not None:
        plt.colorbar(label=colorbal_label)

from typing import Iterable, List

import matplotlib.pyplot as plt

from valuation.utils import Dataset


def plot_iris(
    data: Dataset,
    indices: List[int] = None,
    colors: Iterable = None,
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
        plt.colorbar()

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
        plt.colorbar()

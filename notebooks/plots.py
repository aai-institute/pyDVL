import matplotlib.pyplot as plt
from typing import Iterable, List
from valuation.utils import Dataset


def plot_iris(data: Dataset,
              indices: List[int] = None,
              colors: Iterable = None,
              plot_test: bool = False):
    """Scatter plots for the iris dataset.

    :param data: split Dataset.
    :param indices: subset of data.indices
    :param colors: use with indices to set the color (e.g. to values)
    :param plot_test: plots the points from the test set too.
    """
    if indices is not None:
        xt = data.x_train[indices]
        yt = data.y_train[indices]
    else:
        xt = data.x_train
        yt = data.y_train

    cc = yt.target if colors is None else colors

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    xmin, xmax = data.x_train[data.feature('sepal length (cm)')].min(),\
                 data.x_train[data.feature('sepal length (cm)')].max()
    ymin, ymax = data.x_train[data.feature('sepal width (cm)')].min(),\
                 data.x_train[data.feature('sepal width (cm)')].max()
    xmargin = 0.1 * (xmax - xmin)
    ymargin = 0.1 * (ymax - ymin)
    plt.scatter(xt['sepal length (cm)'], xt['sepal width (cm)'],
                c=cc, marker="1")
    if plot_test:
        plt.scatter(data.x_test[data.feature('sepal length (cm)')],
                    data.x_test[data.feature('sepal width (cm)')],
                    c=data.y_test.target, marker='o', alpha=0.4)
    plt.xlim(xmin - xmargin, xmax + xmargin)
    plt.ylim(ymin - ymargin, ymax + ymargin)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    if colors is not None:
        plt.colorbar()

    plt.subplot(1, 2, 2)
    xmin, xmax = data.x_train['petal length (cm)'].min(),\
                 data.x_train['petal length (cm)'].max()
    ymin, ymax = data.x_train['petal width (cm)'].min(),\
                 data.x_train['petal width (cm)'].max()
    xmargin = 0.1 * (xmax - xmin)
    ymargin = 0.1 * (ymax - ymin)
    plt.scatter(xt['petal length (cm)'], xt['petal width (cm)'],
                c=cc, marker="1")
    if plot_test:
        plt.scatter(data.x_test['petal length (cm)'],
                    data.x_test['petal width (cm)'],
                    c=data.y_test.target, marker='o', alpha=0.4)
    plt.xlim(xmin - xmargin, xmax + xmargin)
    plt.ylim(ymin - ymargin, ymax + ymargin)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    if colors is not None:
        plt.colorbar()

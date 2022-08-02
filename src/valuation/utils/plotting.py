from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_shapley_pydvl(
    dval_df: pd.DataFrame,
    figsize: Tuple[int, int] = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
):
    """Plots the shapley values, as returned from shapley_pydvl.

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


def plot_datasets(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    x_min: np.ndarray,
    x_max: np.ndarray,
    line: np.ndarray,
    colors: Dict[str, np.ndarray] = None,
):

    has_custom_colors = colors is not None
    num_datasets = len(datasets)
    fig, ax = plt.subplots(1, num_datasets, figsize=(12, 4))
    v_max = None
    if has_custom_colors:
        v_max = max([np.max(v) for k, v in colors.items()])

    for i, dataset_name in enumerate(datasets.keys()):
        x, y = datasets[dataset_name]
        ax[i].set_title(dataset_name)
        ax[i].set_xlim(x_min[0], x_max[0])
        ax[i].set_ylim(x_min[1], x_max[1])
        ax[i].plot(line[:, 0], line[:, 1], color="black")

        if not has_custom_colors:
            for v in np.unique(y):
                idx = np.argwhere(y == v)
                ax[i].scatter(x[idx, 0], x[idx, 1], label=str(v))
        else:
            points = ax[i].scatter(
                x[:, 0],
                x[:, 1],
                c=colors[dataset_name],
                vmin=0,
                vmax=v_max,
                cmap="plasma",
            )

    if has_custom_colors:
        plt.colorbar(points)

    plt.legend()
    plt.show()

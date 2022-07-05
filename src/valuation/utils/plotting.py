from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_datasets(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    x_min: np.ndarray = None,
    x_max: np.ndarray = None,
    line: np.ndarray = None,
    s: float = None,
):
    """
    Plots a dictionary of 2-dimensional datasets to either a continuous regression value or a discrete class label.
    In the former case the plasma color map is selected and in the later the tab10 color map is chosen.

    :param datasets: A dictionary mapping dataset names to a tuple of (features, target_variable). Note that the
    features have size [Nx2] and the target_variable [N].
    :param x_min: Set to define the minimum boundaries of the plot.
    :param x_max: Set to define the maximum boundaries of the plot.
    :param line: Optional, line of shape [Mx2], where each row is a point of the 2-dimensional line.
    :parm s: The thickness of the points to plot.
    """

    num_datasets = len(datasets)
    fig, ax = plt.subplots(1, num_datasets, figsize=(12, 4))

    discrete_keys = [
        key for key, dataset in datasets.items() if dataset[1].dtype == int
    ]
    all_discrete_sets = len(discrete_keys) == len(datasets)
    continuous_keys = [key for key in datasets.keys() if key not in discrete_keys]

    v_min = (
        None
        if all_discrete_sets
        else min([np.min(datasets[k][1]) for k in continuous_keys])
    )
    v_max = (
        None
        if all_discrete_sets
        else max([np.max(datasets[k][1]) for k in continuous_keys])
    )
    points = None

    if num_datasets == 1:
        ax = [ax]

    for i, dataset_name in enumerate(datasets.keys()):
        x, y = datasets[dataset_name]
        is_discrete = y.dtype == int

        ax[i].set_title(dataset_name)
        if x_min is not None:
            ax[i].set_xlim(x_min[0], x_max[0])
        if x_max is not None:
            ax[i].set_ylim(x_min[1], x_max[1])

        if line is not None:
            ax[i].plot(line[:, 0], line[:, 1], color="black")

        ret = ax[i].scatter(
            x[:, 0],
            x[:, 1],
            c=y,
            vmin=v_min,
            vmax=v_max,
            cmap="plasma" if not is_discrete else "tab10",
            s=s,
        )
        if not is_discrete:
            points = ret

    if not all_discrete_sets:
        plt.colorbar(points)

    plt.legend()
    plt.show()

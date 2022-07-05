from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_datasets(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    x_min: np.ndarray = None,
    x_max: np.ndarray = None,
    line: np.ndarray = None,
    colors: Dict[str, np.ndarray] = None,
    s: float = None,
):

    has_custom_colors = colors is not None
    num_datasets = len(datasets)
    fig, ax = plt.subplots(1, num_datasets, figsize=(12, 4))
    v_max = None
    if has_custom_colors:
        v_max = max([np.max(v) for k, v in colors.items()])

    if num_datasets == 1:
        ax = [ax]

    for i, dataset_name in enumerate(datasets.keys()):
        x, y = datasets[dataset_name]
        ax[i].set_title(dataset_name)
        if x_min is not None:
            ax[i].set_xlim(x_min[0], x_max[0])
        if x_max is not None:
            ax[i].set_ylim(x_min[1], x_max[1])

        if line is not None:
            ax[i].plot(line[:, 0], line[:, 1], color="black")

        if not has_custom_colors:
            for v in np.unique(y):
                idx = np.argwhere(y == v)
                ax[i].scatter(x[idx, 0], x[idx, 1], label=str(v), s=s)
        else:
            points = ax[i].scatter(
                x[:, 0],
                x[:, 1],
                c=colors[dataset_name],
                vmin=0,
                vmax=v_max,
                cmap="plasma",
                s=s,
            )

    if has_custom_colors:
        plt.colorbar(points)

    plt.legend()
    plt.show()

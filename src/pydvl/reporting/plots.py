from typing import TYPE_CHECKING, Any, List, Optional, OrderedDict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from numpy.typing import NDArray


def shaded_mean_std(
    data: np.ndarray,
    abscissa: Optional[Sequence[Any]] = None,
    num_std: float = 1.0,
    mean_color: Optional[str] = "dodgerblue",
    shade_color: Optional[str] = "lightblue",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:
    """The usual mean +- x std deviations plot to aggregate runs of experiments.

    :param data: axis 0 is to be aggregated on (e.g. runs) and axis 1 is the
        data for each run.
    :param abscissa: values for the x axis. Leave empty to use increasing
        integers.
    :param num_std: number of standard deviations to shade around the mean.
    :param mean_color: color for the mean
    :param shade_color: color for the shaded region
    :param title:
    :param xlabel:
    :param ylabel:
    :param ax: If passed, axes object into which to insert the figure. Otherwise,
        a new figure is created and returned
    :param kwargs: these are forwarded to the ax.plot() call for the mean.

    :return: The axes used (or created)
    """
    assert len(data.shape) == 2
    mean = data.mean(axis=0)
    std = num_std * data.std(axis=0)

    if ax is None:
        fig, ax = plt.subplots()
    if abscissa is None:
        abscissa = list(range(data.shape[1]))

    ax.fill_between(abscissa, mean - std, mean + std, alpha=0.3, color=shade_color)
    ax.plot(abscissa, mean, color=mean_color, **kwargs)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def shapley_results(results: dict, filename: str = None):
    """
    FIXME: change this to use dataframes

    :param results: dict
    :param filename: For plt.savefig(). Set to None to disable saving.

    Here's an example results dictionary::

        results = {
            "all_values": num_runs x num_points
            "backward_scores": num_runs x num_points,
            "backward_scores_reversed": num_runs x num_points,
            "backward_random_scores": num_runs x num_points,
            "forward_scores": num_runs x num_points,
            "forward_scores_reversed": num_runs x num_points,
            "forward_random_scores": num_runs x num_points,
            "max_iterations": int,
            "score_name" str,
            "num_points": int
        }
    """
    plt.figure(figsize=(16, 5))
    num_runs = len(results["all_values"])
    num_points = len(results["backward_scores"][0])
    use_points = int(0.6 * num_points)

    plt.subplot(1, 2, 1)
    values = np.array(results["backward_scores"])[:, :use_points]
    shaded_mean_std(values, color="b", label="By increasing shapley value")

    values = np.array(results["backward_scores_reversed"])[:, :use_points]
    shaded_mean_std(values, color="g", label="By decreasing shapley value")

    values = np.array(results["backward_random_scores"])[:, :use_points]
    shaded_mean_std(values, color="r", linestyle="--", label="At random")

    plt.ylabel(f'Score ({results.get("score_name")})')
    plt.xlabel("Points removed")
    plt.title(
        f"Effect of point removal. "
        f'MonteCarlo with {results.get("max_iterations")} iterations '
        f"over {num_runs} runs"
    )
    plt.legend()

    plt.subplot(1, 2, 2)

    values = np.array(results["forward_scores"])[:, :use_points]
    shaded_mean_std(values, color="b", label="By increasing shapley value")

    values = np.array(results["forward_scores_reversed"])[:, :use_points]
    shaded_mean_std(values, color="g", label="By decreasing shapley value")

    values = np.array(results["forward_random_scores"])[:, :use_points]
    shaded_mean_std(values, color="r", linestyle="--", label="At random")

    plt.ylabel(f'Score ({results.get("score_name")})')
    plt.xlabel("Points added")
    plt.title(
        f"Effect of point addition. "
        f'MonteCarlo with {results["max_iterations"]} iterations '
        f"over {num_runs} runs"
    )
    plt.legend()

    if filename:
        plt.savefig(filename, dpi=300)


def spearman_correlation(vv: List[OrderedDict], num_values: int, pvalue: float):
    """Simple matrix plots with spearman correlation for each pair in vv.

    :param vv: list of OrderedDicts with index: value. Spearman correlation
        is computed for the keys.
    :param num_values: Use only these many values from the data (from the start
        of the OrderedDicts)
    :param pvalue: correlation coefficients for which the p-value is below the
        threshold `pvalue/len(vv)` will be discarded.
    """
    r: np.ndarray = np.ndarray((len(vv), len(vv)))
    p: np.ndarray = np.ndarray((len(vv), len(vv)))
    for i, a in enumerate(vv):
        for j, b in enumerate(vv):
            from scipy.stats._stats_py import SpearmanrResult

            spearman: SpearmanrResult = sp.stats.spearmanr(
                list(a.keys())[:num_values], list(b.keys())[:num_values]
            )
            r[i][j] = (
                spearman.correlation if spearman.pvalue < pvalue / len(vv) else np.nan
            )  # Bonferroni correction
            p[i][j] = spearman.pvalue
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    plot1 = axs[0].matshow(r, vmin=-1, vmax=1)
    axs[0].set_title(f"Spearman correlation (top {num_values} values)")
    axs[0].set_xlabel("Runs")
    axs[0].set_ylabel("Runs")
    fig.colorbar(plot1, ax=axs[0])
    plot2 = axs[1].matshow(p, vmin=0, vmax=1)
    axs[1].set_title("p-value")
    axs[1].set_xlabel("Runs")
    axs[1].set_ylabel("Runs")
    fig.colorbar(plot2, ax=axs[1])

    return fig


def plot_shapley(
    df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
) -> plt.Axes:
    """Plots the shapley values, as returned from shapley.compute_shapley_values.

    :param dval_df: dataframe with the shapley values
    :param figsize: tuple with figure size
    :param title: string, title of the plot
    :param xlabel: string, x label of the plot
    :param ylabel: string, y label of the plot
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.errorbar(
        x=df.index, y=df["data_value"], yerr=df["data_value_std"], fmt="o", capsize=6
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=60)
    return ax


def plot_influence_distribution_by_label(
    influences: "NDArray[np.float_]",
    labels: "NDArray[np.float_]",
    title_extra: str = "",
):
    """Plots the histogram of the influence that all samples in the training set
     have over a single sample index, separated by labels.

    :param influences: array of influences (training samples x test samples)
    :param labels: labels for the training set.
    :param title_extra:
    """
    _, ax = plt.subplots()
    unique_labels = np.unique(labels)
    for label in unique_labels:
        ax.hist(influences[labels == label], label=label, alpha=0.7)
    ax.set_xlabel("Influence values")
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Distribution of influences " + title_extra)
    ax.legend()
    plt.show()

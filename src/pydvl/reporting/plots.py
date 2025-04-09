from __future__ import annotations

from functools import partial
from typing import Any, List, Literal, OrderedDict, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from deprecate import deprecated
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.stats import norm, t

from pydvl.valuation.result import ValuationResult

__all__ = [
    "plot_ci_array",
    "plot_ci_values",
    "plot_result_errors",
    "plot_influence_distribution",
    "plot_influence_distribution_by_label",
    "spearman_correlation",
]


@deprecated(target=None, deprecated_in="0.7.1", remove_in="0.9.0")
def shaded_mean_std(
    data: np.ndarray,
    abscissa: Sequence[Any] | None = None,
    num_std: float = 1.0,
    mean_color: str | None = "dodgerblue",
    shade_color: str | None = "lightblue",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    r"""The usual mean \(\pm\) std deviation plot to aggregate runs of
    experiments.

    !!! warning "Deprecation notice"
        This function is bogus and will be removed in the future in favour of
        properly computed confidence intervals.

    Args:
        data: axis 0 is to be aggregated on (e.g. runs) and axis 1 is the
            data for each run.
        abscissa: values for the x-axis. Leave empty to use increasing integers.
        num_std: number of standard deviations to shade around the mean.
        mean_color: color for the mean
        shade_color: color for the shaded region
        title: Title text. To use mathematics, use LaTeX notation.
        xlabel: Text for the horizontal axis.
        ylabel: Text for the vertical axis
        ax: If passed, axes object into which to insert the figure. Otherwise,
            a new figure is created and returned
        kwargs: these are forwarded to the ax.plot() call for the mean.

    Returns:
        The axes used (or created)
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

    ax.set_title(title or "")
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    ax.set_xticks(abscissa)
    ax.set_xticklabels(abscissa, rotation=60)

    return ax


def plot_ci_array(
    data: NDArray,
    level: float,
    type: Literal["normal", "t", "auto"] = "normal",
    abscissa: Sequence[str] | None = None,
    mean_color: str | None = "dodgerblue",
    shade_color: str | None = "lightblue",
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> plt.Axes:
    """Plot values and a confidence interval from a 2D array.

    Supported intervals are based on the normal and the t distributions.

    Args:
        data: A 2D array with M different values for each of the N indices.
        level: The confidence level.
        type: The type of confidence interval to use.
        abscissa: The values for the x-axis. Leave empty to use increasing
            integers.
        mean_color: The color of the mean line.
        shade_color: The color of the confidence interval.
        ax: If passed, axes object into which to insert the figure. Otherwise,
            a new figure is created and the axes returned.
        **kwargs: Additional arguments to pass to the plot function.

    Returns:
        The matplotlib axes.
    """

    m, n = data.shape

    means = np.mean(data, axis=0)
    variances = np.var(data, axis=0, ddof=1)

    dummy = ValuationResult(
        algorithm="dummy",
        values=means,
        variances=variances,
        counts=np.ones_like(means, dtype=np.int_) * m,
        indices=np.arange(n),
        data_names=(
            np.array(abscissa, dtype=str)
            if abscissa is not None
            else np.arange(n, dtype=str)
        ),
    ).sort(key="index", inplace=True)

    return plot_ci_values(
        dummy,
        level=level,
        type=type,
        abscissa=abscissa,
        mean_color=mean_color,
        shade_color=shade_color,
        ax=ax,
        **kwargs,
    )


def plot_ci_values(
    values: ValuationResult,
    level: float,
    type: Literal["normal", "t", "auto"] = "auto",
    abscissa: Sequence[Any] | None = None,
    mean_color: str | None = "dodgerblue",
    shade_color: str | None = "lightblue",
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> plt.Axes:
    """Plot values and a confidence interval.

    Supported intervals are based on the normal and the t distributions.

    Args:
        values: The valuation result. The object must be sorted by calling
            `ValuationResult.sort()`.
        level: The confidence level.
        type: The type of confidence interval to use. If "auto", uses "norm" if
            the minimum number of updates for all indices is greater than 30,
            otherwise uses "t".
        abscissa: The values for the x-axis. Leave empty to use increasing
            integers.
        mean_color: The color of the mean line.
        shade_color: The color of the confidence interval.
        ax: If passed, axes object into which to insert the figure. Otherwise,
            a new figure is created and the axes returned.
        **kwargs: Additional arguments to pass to the plot function.

    Returns:
        The matplotlib axes.
    """
    assert values._sort_order is not None, "Values must be sorted first."

    ppfs = {
        "normal": norm.ppf,
        "t": partial(t.ppf, df=values.counts - 1),
        "auto": (
            norm.ppf
            if np.min(values.counts) > 30
            else partial(t.ppf, df=values.counts - 1)
        ),
    }

    try:
        score = ppfs[type](1 - level / 2)
    except KeyError:
        raise ValueError(
            f"Unknown confidence interval type requested: {type}."
        ) from None

    if abscissa is None:
        abscissa = range(len(values))

    bound = score * values.stderr

    if ax is None:
        fig, ax = plt.subplots()

    ax.fill_between(
        abscissa,
        values.values - bound,
        values.values + bound,
        alpha=0.3,
        color=shade_color,
    )
    ax.plot(abscissa, values.values, color=mean_color, **kwargs)
    ax.set_xlim(left=min(abscissa), right=max(abscissa))
    return ax


def spearman_correlation(vv: List[OrderedDict], num_values: int, pvalue: float):
    """Simple matrix plots with spearman correlation for each pair in vv.

    Args:
        vv: list of OrderedDicts with index: value. Spearman correlation
            is computed for the keys.
        num_values: Use only these many values from the data (from the start
            of the OrderedDicts)
        pvalue: correlation coefficients for which the p-value is below the
            threshold `pvalue/len(vv)` will be discarded.
    """
    r: np.ndarray = np.ndarray((len(vv), len(vv)))
    p: np.ndarray = np.ndarray((len(vv), len(vv)))
    for i, a in enumerate(vv):
        for j, b in enumerate(vv):
            spearman = sp.stats.spearmanr(
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


def plot_result_errors(
    result: ValuationResult,
    *,
    level: float = 0.05,
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend_label: str | None = None,
) -> plt.Axes:
    r"""Plots a [ValuationResult][pydvl.pydvl.valuation.result.ValuationResult],
    with error bars corresponding to an $\alpha$-level Normal confidence
    interval.

    Args:
        result: object to plot
        level: confidence level for the error bars
        ax: axes to plot on or None if a new subplots should be created
        title: string, title of the plot
        xlabel: string, x label of the plot
        ylabel: string, y label of the plot

    Returns:
        The axes created or used
    """
    if ax is None:
        _, ax = plt.subplots()

    stderr = np.sqrt(result.variances / np.maximum(1.0, result.counts))
    yerr = norm.ppf(1 - level / 2) * stderr

    ax.errorbar(
        x=result.names,
        y=result.values,
        yerr=yerr,
        fmt="o",
        capsize=6,
        label=legend_label,
    )
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    ax.set_title(title or "")
    plt.xticks(rotation=60)
    return ax


def plot_influence_distribution(
    influences: NDArray[np.float64], index: int, title_extra: str = ""
) -> plt.Axes:
    """Plots the histogram of the influence that all samples in the training set
    have over a single sample index.

    Args:
       influences: array of influences (training samples x test samples)
       index: Index of the test sample for which the influences
            will be plotted.
       title_extra: Additional text that will be appended to the title.
    """
    _, ax = plt.subplots()
    ax.hist(influences[:, index], alpha=0.7)
    ax.set_xlabel("Influence values")
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Distribution of influences {title_extra}")
    return cast(plt.Axes, ax)


def plot_influence_distribution_by_label(
    influences: NDArray[np.float64], labels: NDArray[np.float64], title_extra: str = ""
):
    """Plots the histogram of the influence that all samples in the training set
    have over a single sample index, separated by labels.

    Args:
       influences: array of influences (training samples x test samples)
       labels: labels for the training set.
       title_extra: Additional text that will be appended to the title.
    """
    _, ax = plt.subplots()
    unique_labels = np.unique(labels)
    for label in unique_labels:
        ax.hist(influences[labels == label], label=label, alpha=0.7)
    ax.set_xlabel("Influence values")
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Distribution of influences {title_extra}")
    ax.legend()
    plt.show()

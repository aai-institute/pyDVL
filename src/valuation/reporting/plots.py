from typing import List, OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def shaded_mean_std(data: np.ndarray, color: str, num_std: float = 1.0, **kwargs):
    """The usual mean +- x std deviations plot to aggregate runs.

    :param data: axis 0 is to be aggregated on (i.e. runs) and axis 1 is the
    data for each run
    :param color: for matplotlib
    :param num_std: number of standard deviations to shade around the mean
    :param kwargs: these are forwarded to the plt.plot() call for the mean
    """
    assert len(data.shape) == 2
    mean = data.mean(axis=0)
    std = num_std * data.std(axis=0)

    plt.fill_between(
        list(range(data.shape[1])), mean - std, mean + std, alpha=0.3, color=color
    )
    plt.plot(mean, color=color, **kwargs)


def shapley_results(results: dict, filename: str = None):
    """
    :param results: dict
        results = {'all_values': num_runs x num_points
                   'backward_scores': num_runs x num_points,
                   'backward_scores_reversed': num_runs x num_points,
                   'backward_random_scores': num_runs x num_points,
                   'forward_scores': num_runs x num_points,
                   'forward_scores_reversed': num_runs x num_points,
                   'forward_random_scores': num_runs x num_points,
                   'max_iterations': int,
                   'score_name: str,
                   'num_points': int}
    :param filename: For plt.savefig(). Set to None to disable saving.
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


def spearman_correlation(vv: List[OrderedDict], num_values: int, pvalue):
    """Simple matrix plots with spearman correlation for each pair in vv.

    :param vv: list of OrderedDicts with index: value. Spearman correlation
               is computed for the keys.
    :param num_values: Use only these many values from the data (from the start
                       of the OrderedDicts)
    """
    r = np.ndarray((len(vv), len(vv)))
    p = np.ndarray((len(vv), len(vv)))
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
    plot1 = axs[0].matshow(r)
    axs[0].set_title(f"Spearman correlation (top {num_values} values)")
    axs[0].set_xlabel("Runs")
    axs[0].set_ylabel("Runs")
    fig.colorbar(plot1, ax=axs[0])
    plot2 = axs[1].matshow(p)
    axs[1].set_title("p-value")
    axs[1].set_xlabel("Runs")
    axs[1].set_ylabel("Runs")
    fig.colorbar(plot2, ax=axs[1])

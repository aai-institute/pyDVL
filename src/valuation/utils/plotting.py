from typing import Tuple

import matplotlib.pyplot as plt
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

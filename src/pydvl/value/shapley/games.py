"""
This module provides several predefined games and their Shapley values, for
benchmarking purposes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy as sp
from numpy.typing import NDArray

from pydvl.utils import Scorer, Status
from pydvl.utils.dataset import Dataset
from pydvl.utils.types import SupervisedModel
from pydvl.utils.utility import Utility
from pydvl.value import ValuationResult

__all__ = [
    "symmetric_voting_game",
    "asymmetric_voting_game",
    "shoes_game",
    "airport_game",
    "minimum_spanning_tree_game",
    "SolvedGame",
]


@dataclass
class SolvedGame:
    u: Utility
    values: ValuationResult


def _dummy_dataset(num_samples: int, description: str) -> Dataset:
    x = np.arange(0, num_samples, 1).reshape(-1, 1)
    nil = np.zeros_like(x)
    return Dataset(
        x,
        nil.copy(),
        nil.copy(),
        nil.copy(),
        feature_names=["x"],
        target_names=["y"],
        description=description,
    )


class DummyModel(SupervisedModel):
    def __init__(self):
        pass

    def fit(self, x: NDArray, y: NDArray):
        pass

    def predict(self, x: NDArray) -> NDArray:
        return x


def symmetric_voting_game(num_samples: int = 1000) -> SolvedGame:
    """A symmetric voting game defined in :footcite:t:`castro_polynomial_2009`
    Section 4.1

    Under this model the utility of a coalition is 1 if its cardinality is
    greater than num_samples/2, or 0 otherwise.
    """
    if num_samples % 2 != 0:
        raise ValueError("num_samples must be an even number.")

    data = _dummy_dataset(
        num_samples, "Dummy data for the symmetric voting game in Castro " "et al. 2009"
    )

    def symmetric_voting_score(model: SupervisedModel, x: NDArray, y: NDArray) -> float:
        return 1 if len(x) > len(data) // 2 else 0

    u = Utility(
        DummyModel(),
        data,
        scorer=Scorer(symmetric_voting_score, range=(0, 1)),
        catch_errors=False,
        show_warnings=True,
        enable_cache=False,
    )
    values = ValuationResult(
        algorithm="exact_shapley",
        status=Status.Converged,
        indices=data.indices,
        values=np.ones_like(data.x_train) / len(data.x_train),
        variances=np.zeros_like(data.x_train),
        counts=np.zeros_like(data.x_train),
    )

    return SolvedGame(u, values)


def asymmetric_voting_game() -> SolvedGame:
    """An asymmetric voting game defined in :footcite:t:`castro_polynomial_2009`
    Section 4.2.
    """
    n = 51
    ranges = [
        range(0, 1),
        range(1, 2),
        range(2, 3),
        range(3, 5),
        range(5, 6),
        range(6, 7),
        range(7, 9),
        range(9, 10),
        range(10, 12),
        range(12, 15),
        range(15, 16),
        range(16, 20),
        range(20, 24),
        range(24, 26),
        range(26, 30),
        range(30, 34),
        range(34, 35),
        range(35, 44),
        range(44, 51),
    ]

    ranges_weights = [
        45,
        41,
        27,
        26,
        25,
        21,
        17,
        14,
        13,
        12,
        11,
        10,
        9,
        8,
        7,
        6,
        5,
        4,
        3,
    ]
    ranges_values = [
        "0.08831",
        "0.07973",
        "0.05096",
        "0.04898",
        "0.047",
        "0.03917",
        "0.03147",
        "0.02577",
        "0.02388",
        "0.022",
        "0.02013",
        "0.01827",
        "0.01641",
        "0.01456",
        "0.01272",
        "0.01088",
        "0.009053",
        "0.00723",
        "0.005412",
    ]

    weight_table = np.zeros(n)
    exact_values = np.zeros(n)
    for r, w, v in zip(ranges, ranges_weights, ranges_values):
        weight_table[r] = w
        exact_values[r] = v

    threshold = np.sum(weight_table) / 2

    def assymetric_voting_game_score(
        model: SupervisedModel, x: NDArray, y: NDArray
    ) -> float:
        return 1 if np.sum(weight_table[x]) > threshold else 0

    data = _dummy_dataset(
        n, "Dummy data for the asymmetric voting game in Castro et al. 2009"
    )

    u = Utility(
        model=DummyModel(),
        data=data,
        scorer=Scorer(assymetric_voting_game_score, range=(0, 1)),
        catch_errors=False,
        show_warnings=True,
        enable_cache=False,
    )

    values = ValuationResult(
        algorithm="exact_shapley",
        status=Status.Converged,
        indices=data.indices,
        values=exact_values,
        variances=np.zeros_like(data.x_train),
        counts=np.zeros_like(data.x_train),
    )

    return SolvedGame(u, values)


def shoes_game(num_samples: int = 1000) -> SolvedGame:
    """A shoes game defined in :footcite:t:`castro_polynomial_2009`

    The utility of a coalition is the minimum of the number of left shoes or
    right shoes in a coalition. A player is a left shoe iff its index is among
    the first half, or a right shoe otherwise.
    """
    if num_samples % 2 != 0:
        raise ValueError("num_samples must be an even number.")

    data = _dummy_dataset(
        num_samples, "Dummy data for the shoe game in Castro et al. 2009"
    )

    m = len(data) // 2

    def shoe_game_score(model: SupervisedModel, x: NDArray, y: NDArray) -> float:
        left_shoes = np.sum(x < m).item()
        right_shoes = np.sum(x >= m).item()
        return min(left_shoes, right_shoes)

    u = Utility(
        model=DummyModel(),
        data=data,
        scorer=Scorer(shoe_game_score, range=(0, m)),
        catch_errors=False,
        show_warnings=True,
        enable_cache=False,
    )

    values = ValuationResult(
        algorithm="exact_shapley",
        status=Status.Converged,
        indices=data.indices,
        values=np.ones_like(data.x_train) * 0.5,
        variances=np.zeros_like(data.x_train),
        counts=np.zeros_like(data.x_train),
    )

    return SolvedGame(u, values)


def airport_game() -> SolvedGame:
    """An airport game defined in :footcite:t:`castro_polynomial_2009`,
    Section 4.3"""
    ranges = [
        range(0, 8),
        range(8, 20),
        range(20, 26),
        range(26, 40),
        range(40, 48),
        range(48, 57),
        range(57, 70),
        range(70, 80),
        range(80, 90),
        range(90, 100),
    ]
    exact = [
        0.01,
        0.020869565,
        0.033369565,
        0.046883079,
        0.063549745,
        0.082780515,
        0.106036329,
        0.139369662,
        0.189369662,
        0.289369662,
    ]
    c = list(range(1, 10))
    score_table = np.zeros(100)
    exact_values = np.zeros(100)

    for r, v in zip(ranges, exact):
        score_table[r] = c
        exact_values[r] = v

    def airport_game_score(model: SupervisedModel, x: NDArray, y: NDArray) -> float:
        return max(score_table[x])

    data = _dummy_dataset(100, "A dummy dataset for...")

    u = Utility(
        model=DummyModel(),
        data=data,
        scorer=Scorer(airport_game_score, range=(0, 100)),
        catch_errors=False,
        show_warnings=True,
        enable_cache=False,
    )

    values = ValuationResult(
        algorithm="exact_shapley",
        status=Status.Converged,
        indices=data.indices,
        values=exact_values,
        variances=np.zeros_like(data.x_train),
        counts=np.zeros_like(data.x_train),
    )

    return SolvedGame(u, values)


def minimum_spanning_tree_game() -> SolvedGame:
    data = _dummy_dataset(101, "A dummy dataset for...")
    n = 101
    graph = np.zeros(shape=(n, n))

    for i in range(n):
        for j in range(n):
            if (
                i == j + 1
                or i == j - 1
                or (i == 1 and j == n - 1)
                or (i == n - 1 and j == 1)
            ):
                graph[i, j] = 1
            elif i == 0 or j == 0:
                graph[i, j] = 0
            else:
                graph[i, j] = np.inf
    assert np.all(graph == graph.T)

    def minimum_spanning_tree_score(
        model: SupervisedModel, x: NDArray, y: NDArray
    ) -> float:
        partial_graph = sp.sparse.csr_array(graph[np.ix_(x, x)])
        span_tree = sp.sparse.csgraph.minimum_spanning_tree(partial_graph)
        return span_tree.sum()

    u = Utility(
        model=DummyModel(),
        data=data,
        scorer=Scorer(minimum_spanning_tree_score, range=(0, np.inf)),
        catch_errors=False,
        show_warnings=True,
        enable_cache=False,
    )

    values = ValuationResult(
        algorithm="exact_shapley",
        status=Status.Converged,
        indices=data.indices,
        values=2 * np.ones_like(data.x_train),
        variances=np.zeros_like(data.x_train),
        counts=np.zeros_like(data.x_train),
    )

    return SolvedGame(u, values)

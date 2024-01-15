"""
This module provides several predefined games and their Shapley values, for
benchmarking purposes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Iterable, Optional, Tuple

import numpy as np
import scipy as sp
from numpy.typing import NDArray

from pydvl.utils import Scorer, Status
from pydvl.utils.dataset import Dataset
from pydvl.utils.types import SupervisedModel
from pydvl.utils.utility import Utility
from pydvl.value import ValuationResult

__all__ = [
    "Game",
    "SymmetricVotingGame",
    "AsymmetricVotingGame",
    "ShoesGame",
    "AirportGame",
    "MinimumSpanningTreeGame",
]


class DummyGameDataset(Dataset):
    def __init__(self, n_players: int, description: Optional[str] = None) -> None:
        x = np.arange(0, n_players, 1).reshape(-1, 1)
        nil = np.zeros_like(x)
        super().__init__(
            x,
            nil.copy(),
            nil.copy(),
            nil.copy(),
            feature_names=["x"],
            target_names=["y"],
            description=description,
        )

    def get_test_data(
        self, indices: Optional[Iterable[int]] = None
    ) -> Tuple[NDArray, NDArray]:
        """Returns the subsets of the train set instead of the test set.

        Args:
            indices: Indices into the traing data.

        Returns:
            Subset of the train data.
        """
        if indices is None:
            return self.x_train, self.y_train
        x = self.x_train[indices]
        y = self.y_train[indices]
        return x, y


class DummyModel(SupervisedModel):
    def __init__(self):
        pass

    def fit(self, x: NDArray, y: NDArray):
        pass

    def predict(self, x: NDArray) -> NDArray:
        pass

    def score(self, x: NDArray, y: NDArray) -> float:
        # Dummy, will be overriden
        return 0


class Game(ABC):
    """Base class for games

    Any Game subclass has to implement the abstract `_score` method
    to assign a score to each coalition/subset and at least
    one of `shapley_values`, `least_core_values`.
    """

    def __init__(
        self,
        n_players: int,
        score_range: Tuple[float, float] = (-np.inf, np.inf),
        description: Optional[str] = None,
    ):
        self.n_players = n_players
        self.data = DummyGameDataset(self.n_players, description)
        self.u = Utility(
            DummyModel(),
            self.data,
            scorer=Scorer(self._score, range=score_range),
            catch_errors=False,
            show_warnings=True,
        )

    def shapley_values(self) -> ValuationResult:
        raise NotImplementedError(
            f"shapley_values method was not implemented for class {__class__.__name__}"
        )

    def least_core_values(self) -> ValuationResult:
        raise NotImplementedError(
            f"least_core_values method was not implemented for class {__class__.__name__}"
        )

    @abstractmethod
    def _score(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_players={self.n_players})"


class SymmetricVotingGame(Game):
    r"""Toy game that is used for testing and demonstration purposes.

    A symmetric voting game defined in :footcite:t:`castro_polynomial_2009`
    Section 4.1

    Under this model the utility of a coalition is 1 if its cardinality is
    greater than num_samples/2, or 0 otherwise.
    """

    def __init__(self, n_players: int) -> None:
        if n_players % 2 != 0:
            raise ValueError("n_players must be an even number.")
        description = "Dummy data for the symmetric voting game in Castro et al. 2009"
        super().__init__(
            n_players,
            score_range=(0, 1),
            description=description,
        )

    def _score(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
        return 1 if len(X) > len(self.data) // 2 else 0

    @lru_cache
    def shapley_values(self) -> ValuationResult:
        exact_values = np.ones_like(self.data.x_train) / len(self.data.x_train)
        result: ValuationResult[np.int_, int] = ValuationResult(
            algorithm="exact_shapley",
            status=Status.Converged,
            indices=self.data.indices,
            values=exact_values,
            variances=np.zeros_like(self.data.x_train),
            counts=np.zeros_like(self.data.x_train),
        )
        return result


class AsymmetricVotingGame(Game):
    """Toy game that is used for testing and demonstration purposes.

    An asymmetric voting game defined in :footcite:t:`castro_polynomial_2009`
    Section 4.2.
    """

    def __init__(self, n_players: int = 51) -> None:
        if n_players != 51:
            raise ValueError(
                f"{__class__.__name__} only supports n_players=51 but got {n_players=}."
            )
        description = "Dummy data for the asymmetric voting game in Castro et al. 2009"
        super().__init__(
            n_players,
            score_range=(0, 1),
            description=description,
        )

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

        self.weight_table = np.zeros(self.n_players)
        exact_values = np.zeros(self.n_players)
        for r, w, v in zip(ranges, ranges_weights, ranges_values):
            self.weight_table[r] = w
            exact_values[r] = v

        self.exact_values = exact_values
        self.threshold = np.sum(self.weight_table) / 2

    def _score(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
        return 1 if np.sum(self.weight_table[X]) > self.threshold else 0

    @lru_cache
    def shapley_values(self) -> ValuationResult:
        result: ValuationResult[np.int_, int] = ValuationResult(
            algorithm="exact_shapley",
            status=Status.Converged,
            indices=self.data.indices,
            values=self.exact_values,
            variances=np.zeros_like(self.data.x_train),
            counts=np.zeros_like(self.data.x_train),
        )
        return result


class ShoesGame(Game):
    """Toy game that is used for testing and demonstration purposes.

    A shoes game defined in :footcite:t:`castro_polynomial_2009`

    The utility of a coalition is the minimum of the number of left shoes or
    right shoes in a coalition. A player is a left shoe iff its index is among
    the first half, or a right shoe otherwise.
    """

    def __init__(self, n_players: int) -> None:
        if n_players % 2 != 0:
            raise ValueError("n_players must be an even number.")
        description = "Dummy data for the shoe game in Castro et al. 2009"
        self.m = n_players // 2
        super().__init__(n_players, score_range=(0, self.m), description=description)

    def _score(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
        left_shoes = np.sum(X < self.m).item()
        right_shoes = np.sum(X >= self.m).item()
        return min(left_shoes, right_shoes)

    @lru_cache
    def shapley_values(self) -> ValuationResult:
        exact_values = np.ones_like(self.data.x_train) * 0.5
        result: ValuationResult[np.int_, int] = ValuationResult(
            algorithm="exact_shapley",
            status=Status.Converged,
            indices=self.data.indices,
            values=exact_values,
            variances=np.zeros_like(self.data.x_train),
            counts=np.zeros_like(self.data.x_train),
        )
        return result


class AirportGame(Game):
    """Toy game that is used for testing and demonstration purposes.

    An airport game defined in :footcite:t:`castro_polynomial_2009`,
    Section 4.3"""

    def __init__(self, n_players: int = 100) -> None:
        if n_players != 100:
            raise ValueError(
                f"{__class__.__name__} only supports n_players=100 but got {n_players=}."
            )
        description = "A dummy dataset for the airport game in Castro et al. 2009"
        super().__init__(n_players, score_range=(0, 100), description=description)
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

        self.exact_values = exact_values
        self.score_table = score_table

    def _score(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
        return max(self.score_table[X]) or 0.0

    @lru_cache
    def shapley_values(self) -> ValuationResult:
        result: ValuationResult[np.int_, int] = ValuationResult(
            algorithm="exact_shapley",
            status=Status.Converged,
            indices=self.data.indices,
            values=self.exact_values,
            variances=np.zeros_like(self.data.x_train),
            counts=np.zeros_like(self.data.x_train),
        )
        return result


class MinimumSpanningTreeGame(Game):
    """Toy game that is used for testing and demonstration purposes."""

    def __init__(self, n_players: int = 101) -> None:
        if n_players != 101:
            raise ValueError(
                f"{__class__.__name__} only supports n_players=101 but got {n_players=}."
            )
        description = (
            "A dummy dataset for the minimum spanning tree game in Castro et al. 2009"
        )
        super().__init__(n_players, score_range=(0, np.inf), description=description)

        graph = np.zeros(shape=(self.n_players, self.n_players))

        for i in range(self.n_players):
            for j in range(self.n_players):
                if (
                    i == j + 1
                    or i == j - 1
                    or (i == 1 and j == self.n_players - 1)
                    or (i == self.n_players - 1 and j == 1)
                ):
                    graph[i, j] = 1
                elif i == 0 or j == 0:
                    graph[i, j] = 0
                else:
                    graph[i, j] = np.inf
        assert np.all(graph == graph.T)

        self.graph = graph

    def _score(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
        partial_graph = sp.sparse.csr_array(self.graph[np.ix_(X, X)])
        span_tree = sp.sparse.csgraph.minimum_spanning_tree(partial_graph)
        return span_tree.sum() or 0

    @lru_cache
    def shapley_values(self) -> ValuationResult:
        exact_values = 2 * np.ones_like(self.data.x_train)
        result: ValuationResult[np.int_, int] = ValuationResult(
            algorithm="exact_shapley",
            status=Status.Converged,
            indices=self.data.indices,
            values=exact_values,
            variances=np.zeros_like(self.data.x_train),
            counts=np.zeros_like(self.data.x_train),
        )
        return result

"""
This module provides several predefined games and, depending on the game,
the corresponding Shapley values, Least Core values or both of them, for
benchmarking purposes.

## References

[^1]: <a name="castro_polynomial_2009"></a>Castro, J., Gómez, D. and Tejada,
      J., 2009. [Polynomial calculation of the Shapley value based on
      sampling](http://www.sciencedirect.com/science/article/pii/S0305054808000804).
      Computers & Operations Research, 36(5), pp.1726-1730.

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
    "MinerGame",
]


class DummyGameDataset(Dataset):
    """Dummy game dataset.

    Initializes a dummy game dataset with n_players and an optional
    description.

    This class is used internally inside the [Game][pydvl.value.games.Game]
    class.

    Args:
        n_players: Number of players that participate in the game.
        description: Optional description of the dataset.
    """

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
            indices: Indices into the training data.

        Returns:
            Subset of the train data.
        """
        if indices is None:
            return self.x_train, self.y_train
        x = self.x_train[indices]
        y = self.y_train[indices]
        return x, y


class DummyModel(SupervisedModel):
    """Dummy model class.

    A dummy supervised model used for testing purposes only.
    """

    def __init__(self) -> None:
        pass

    def fit(self, x: NDArray, y: NDArray | None) -> None:
        pass

    def predict(self, x: NDArray) -> NDArray:  # type: ignore
        pass

    def score(self, x: NDArray, y: NDArray | None) -> float:
        # Dummy, will be overriden
        return 0


class Game(ABC):
    """Base class for games

    Any Game subclass has to implement the abstract `_score` method
    to assign a score to each coalition/subset and at least
    one of `shapley_values`, `least_core_values`.

    Args:
        n_players: Number of players that participate in the game.
        score_range: Minimum and maximum values of the `_score` method.
        description: Optional string description of the dummy dataset that will be created.

    Attributes:
        n_players: Number of players that participate in the game.
        data: Dummy dataset object.
        u: Utility object with a dummy model and dataset.
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
            f"shapley_values method was not implemented for class {self.__class__.__name__}"
        )

    def least_core_values(self) -> ValuationResult:
        raise NotImplementedError(
            f"least_core_values method was not implemented for class {self.__class__.__name__}"
        )

    @abstractmethod
    def _score(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_players={self.n_players})"


class SymmetricVotingGame(Game):
    r"""Toy game that is used for testing and demonstration purposes.

    A symmetric voting game defined in
    (Castro et al., 2009)<sup><a href="#castro_polynomial_2009">1</a></sup>
    Section 4.1

    For this game the utility of a coalition is 1 if its cardinality is
    greater than num_samples/2, or 0 otherwise.

    $${
    v(S) = \left\{\begin{array}{ll}
    1, & \text{ if} \quad \mid S \mid > \frac{N}{2} \\
    0, & \text{ otherwise}
    \end{array}\right.
    }$$

    Args:
        n_players: Number of players that participate in the game.
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
        exact_values = np.ones(self.n_players) / self.n_players
        result: ValuationResult[np.int_, np.int_] = ValuationResult(
            algorithm="exact_shapley",
            status=Status.Converged,
            indices=self.data.indices,
            values=exact_values,
            variances=np.zeros_like(self.data.x_train),
            counts=np.zeros_like(self.data.x_train),
        )
        return result


class AsymmetricVotingGame(Game):
    r"""Toy game that is used for testing and demonstration purposes.

    An asymmetric voting game defined in
    (Castro et al., 2009)<sup><a href="#castro_polynomial_2009">1</a></sup>
    Section 4.2.

    For this game the player set is $N = \{1,\dots,51\}$ and
    the utility of a coalition is given by:

    $${
    v(S) = \left\{\begin{array}{ll}
    1, & \text{ if} \quad \sum\limits_{i \in S} w_i > \sum\limits_{j \in N}\frac{w_j}{2} \\
    0, & \text{ otherwise}
    \end{array}\right.
    }$$

    where $w = [w_1,\dots, w_{51}]$ is a list of weights associated with each player.

    Args:
        n_players: Number of players that participate in the game.
    """

    def __init__(self, n_players: int = 51) -> None:
        if n_players != 51:
            raise ValueError(
                f"{self.__class__.__name__} only supports n_players=51 but got {n_players=}."
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
        result: ValuationResult[np.int_, np.int_] = ValuationResult(
            algorithm="exact_shapley",
            status=Status.Converged,
            indices=self.data.indices,
            values=self.exact_values,
            variances=np.zeros_like(self.data.x_train),
            counts=np.zeros_like(self.data.x_train),
        )
        return result


class ShoesGame(Game):
    r"""Toy game that is used for testing and demonstration purposes.

    A shoes game defined in (Castro et al.,
    2009)<sup><a href="#castro_polynomial_2009">1</a></sup>.

    In this game, some players have a left shoe and others a right shoe.
    Single shoes have a worth of zero while pairs have a worth of 1.

    The payoff of a coalition $S$ is:

    $${
    v(S) = \min( \mid S \cap L \mid, \mid S \cap R \mid )
    }$$

    Where $L$, respectively $R$, is the set of players with left shoes,
    respectively right shoes.

    Args:
        left: Number of players with a left shoe.
        right: Number of players with a right shoe.
    """

    def __init__(self, left: int, right: int) -> None:
        self.left = left
        self.right = right
        n_players = self.left + self.right
        description = "Dummy data for the shoe game in Castro et al. 2009"
        max_score = n_players // 2
        super().__init__(n_players, score_range=(0, max_score), description=description)

    def _score(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
        left_sum = float(np.sum(np.asarray(X) < self.left))
        right_sum = float(np.sum(np.asarray(X) >= self.left))
        return min(left_sum, right_sum)

    @lru_cache
    def shapley_values(self) -> ValuationResult:
        if self.left != self.right and (self.left > 4 or self.right > 4):
            raise ValueError(
                "This class only supports getting exact shapley values "
                "for left <= 4 and right <= 4 or left == right"
            )
        precomputed_values = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.667, 0.75, 0.8],
                [0.0, 0.167, 0.5, 0.65, 0.733],
                [0.0, 0.083, 0.233, 0.5, 0.638],
                [0.0, 0.050, 0.133, 0.271, 0.5],
            ]
        )
        if self.left == self.right:
            value_left = value_right = min(self.left, self.right) / (
                self.left + self.right
            )
        else:
            value_left = precomputed_values[self.left, self.right]
            value_right = precomputed_values[self.right, self.left]
        exact_values = np.array([value_left] * self.left + [value_right] * self.right)
        result: ValuationResult[np.int_, np.int_] = ValuationResult(
            algorithm="exact_shapley",
            status=Status.Converged,
            indices=self.data.indices,
            values=exact_values,
            variances=np.zeros_like(self.data.x_train),
            counts=np.zeros_like(self.data.x_train),
        )
        return result

    @lru_cache
    def least_core_values(self) -> ValuationResult:
        if self.left == self.right:
            subsidy = -0.5
            exact_values = np.array([0.5] * (self.left + self.right))
        elif self.left < self.right:
            subsidy = 0.0
            exact_values = np.array([1.0] * self.left + [0.0] * self.right)
        else:
            subsidy = 0.0
            exact_values = np.array([0.0] * self.left + [1.0] * self.right)

        result: ValuationResult[np.int_, np.int_] = ValuationResult(
            algorithm="exact_least_core",
            status=Status.Converged,
            indices=self.data.indices,
            values=exact_values,
            subsidy=subsidy,
            variances=np.zeros_like(self.data.x_train),
            counts=np.zeros_like(self.data.x_train),
        )
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(L={self.left}, R={self.right})"


class AirportGame(Game):
    """Toy game that is used for testing and demonstration purposes.

    An airport game defined in
    (Castro et al., 2009)<sup><a href="#castro_polynomial_2009">1</a></sup>
    Section 4.3

    Args:
        n_players: Number of players that participate in the game.
    """

    def __init__(self, n_players: int = 100) -> None:
        if n_players != 100:
            raise ValueError(
                f"{self.__class__.__name__} only supports n_players=100 but got {n_players=}."
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
        result: ValuationResult[np.int_, np.int_] = ValuationResult(
            algorithm="exact_shapley",
            status=Status.Converged,
            indices=self.data.indices,
            values=self.exact_values,
            variances=np.zeros_like(self.data.x_train),
            counts=np.zeros_like(self.data.x_train),
        )
        return result


class MinimumSpanningTreeGame(Game):
    r"""Toy game that is used for testing and demonstration purposes.

    A minimum spanning tree game defined in
    (Castro et al., 2009)<sup><a href="#castro_polynomial_2009">1</a></sup>.

    Let $G = (N \cup \{0\},E)$ be a valued graph where $N = \{1,\dots,100\}$,
    and the cost associated to an edge $(i, j)$ is:

    $${
    c_{ij} = \left\{\begin{array}{lll}
    1, & \text{ if} & i = j + 1 \text{ or } i = j - 1 \\
    & & \text{ or } (i = 1 \text{ and } j = 100) \text{ or } (i = 100 \text{ and } j = 1) \\
    101, & \text{ if} & i = 0 \text{ or } j = 0 \\
    \infty, & \text{ otherwise}
    \end{array}\right.
    }$$

    A minimum spanning tree game $(N, c)$ is a cost game, where for a given coalition
    $S \subset N$, $v(S)$ is the sum of the edge cost of the minimum spanning tree,
    i.e. $v(S)$ = Minimum Spanning Tree of the graph $G|_{S\cup\{0\}}$,
    which is the partial graph restricted to the players $S$ and the source node $0$.

    Args:
        n_players: Number of players that participate in the game.
    """

    def __init__(self, n_players: int = 100) -> None:
        if n_players != 100:
            raise ValueError(
                f"{self.__class__.__name__} only supports n_players=100 but got {n_players=}."
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
        result: ValuationResult[np.int_, np.int_] = ValuationResult(
            algorithm="exact_shapley",
            status=Status.Converged,
            indices=self.data.indices,
            values=exact_values,
            variances=np.zeros_like(self.data.x_train),
            counts=np.zeros_like(self.data.x_train),
        )
        return result


class MinerGame(Game):
    r"""Toy game that is used for testing and demonstration purposes.

    Consider a group of n miners, who have discovered large bars of gold.

    If two miners can carry one piece of gold, then the payoff of a
    coalition $S$ is:

    $${
    v(S) = \left\{\begin{array}{lll}
    \mid S \mid / 2, & \text{ if} & \mid S \mid \text{ is even} \\
    ( \mid S \mid - 1)/2, & \text{ otherwise}
    \end{array}\right.
    }$$

    If there are more than two miners and there is an even number of miners,
    then the core consists of the single payoff where each miner gets 1/2.

    If there is an odd number of miners, then the core is empty.

    Taken from [Wikipedia](https://en.wikipedia.org/wiki/Core_(game_theory))

    Args:
        n_players: Number of miners that participate in the game.
    """

    def __init__(self, n_players: int) -> None:
        if n_players <= 2:
            raise ValueError(f"n_players, {n_players}, should be > 2")
        description = "Dummy data for Miner Game taken from https://en.wikipedia.org/wiki/Core_(game_theory)"
        super().__init__(
            n_players,
            score_range=(0, n_players // 2),
            description=description,
        )

    def _score(self, model: SupervisedModel, X: NDArray, y: NDArray) -> float:
        n = len(X)
        if n % 2 == 0:
            return n / 2
        else:
            return (n - 1) / 2

    @lru_cache()
    def least_core_values(self) -> ValuationResult:
        if self.n_players % 2 == 0:
            values = np.array([0.5] * self.n_players)
            subsidy = 0.0
        else:
            values = np.array(
                [(self.n_players - 1) / (2 * self.n_players)] * self.n_players
            )
            subsidy = (self.n_players - 1) / (2 * self.n_players)

        result: ValuationResult[np.int_, np.int_] = ValuationResult(
            algorithm="exact_least_core",
            status=Status.Converged,
            indices=self.data.indices,
            values=values,
            subsidy=subsidy,
            variances=np.zeros_like(self.data.x_train),
            counts=np.zeros_like(self.data.x_train),
        )
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={self.n_players})"

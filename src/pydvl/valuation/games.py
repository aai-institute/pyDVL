"""
This module provides several predefined games used in the literature [^1] and, depending
on the game, precomputed Shapley, Least-Core, and / or Banzhaf values, for benchmarking
purposes.

The games are:

- [SymmetricVotingGame][pydvl.valuation.games.SymmetricVotingGame]
- [AsymmetricVotingGame][pydvl.valuation.games.AsymmetricVotingGame]
- [ShoesGame][pydvl.valuation.games.ShoesGame]
- [AirportGame][pydvl.valuation.games.AirportGame]
- [MinimumSpanningTreeGame][pydvl.valuation.games.MinimumSpanningTreeGame]
- [MinerGame][pydvl.valuation.games.MinerGame]


## References

[^1]: <a name="castro_polynomial_2009"></a>Castro, J., Gómez, D. and Tejada,
      J., 2009. [Polynomial calculation of the Shapley value based on
      sampling](http://www.sciencedirect.com/science/article/pii/S0305054808000804).
      Computers & Operations Research, 36(5), pp.1726-1730.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Callable

import numpy as np
import scipy as sp
from numpy.typing import NDArray

from pydvl.utils.status import Status
from pydvl.valuation.dataset import Dataset
from pydvl.valuation.methods._solve_least_core_problems import LeastCoreProblem
from pydvl.valuation.result import ValuationResult
from pydvl.valuation.types import SampleT, SupervisedModel
from pydvl.valuation.utility.base import UtilityBase

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

    Initializes a dummy game dataset with `n_players` and an optional description.

    This class is used internally inside the [Game][pydvl.valuation.games.Game] class.

    Args:
        n_players: Number of players that participate in the game.
        description: Optional description of the dataset.
    """

    def __init__(self, n_players: int, description: str | None = None) -> None:
        x = np.arange(0, n_players, 1).reshape(-1, 1)
        nil = np.zeros_like(x)
        super().__init__(
            x,
            nil.copy(),
            feature_names=["x"],
            target_names=["y"],
            description=description,
        )


class DummyGameUtility(UtilityBase):
    """Dummy game utility

    This class is used internally inside the [Game][pydvl.valuation.games.Game] class.

    Args:
        score: Function to compute the score of a coalition.
        score_range: Minimum and maximum values of the score function.
    """

    def __init__(
        self, score: Callable[[NDArray], float], score_range: tuple[float, float]
    ):
        self.score = score
        self.score_range = score_range

    def __call__(self, sample: SampleT | None) -> float:
        if sample is None or len(sample.subset) == 0:
            return 0

        if self.training_data is None:
            raise ValueError("Utility object has no training data.")

        idxs: NDArray[np.int_] = np.array(sample.subset, dtype=np.int_)
        x, _ = self.training_data.data(idxs)
        try:
            score: float = self.score(x)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            score = 0.0
        return score

    def with_dataset(self, dataset: Dataset, copy: bool = True):
        utility = type(self)(self.score, self.score_range) if copy else self
        utility._training_data = dataset
        return utility


class DummyModel(SupervisedModel[NDArray, NDArray]):
    """Dummy model class.

    A dummy supervised model used for testing purposes only.
    """

    def __init__(self):
        pass

    def fit(self, x: NDArray, y: NDArray | None):
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
    one of `shapley_values`, `least_core_values`, or `banzhaf_values`.

    Args:
        n_players: Number of players that participate in the game.
        score_range: Minimum and maximum values of the `_score` method.

    Attributes:
        n_players: Number of players that participate in the game.
        data: Dummy dataset object.
        u: Utility object with a dummy model and dataset.
    """

    def __init__(
        self,
        n_players: int,
        score_range: tuple[float, float] = (-np.inf, np.inf),
    ):
        self.n_players = n_players
        self.data = DummyGameDataset(
            n_players=self.n_players,
            description=f"Dummy data for {self.__class__.__name__}",
        )
        self.u = DummyGameUtility(score=self._score, score_range=score_range)

    def shapley_values(self) -> ValuationResult:
        raise NotImplementedError(
            f"shapley_values method was not implemented for class {self.__class__.__name__}"
        )

    def least_core_values(self) -> ValuationResult:
        raise NotImplementedError(
            f"least_core_values method was not implemented for class {self.__class__.__name__}"
        )

    def banzhaf_values(self) -> ValuationResult:
        raise NotImplementedError(
            f"banzhaf_values method was not implemented for class {self.__class__.__name__}"
        )

    @abstractmethod
    def _score(self, X: NDArray) -> float: ...

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

    def __init__(self, n_players: int):
        if n_players % 2 != 0:
            raise ValueError("n_players must be an even number.")
        super().__init__(n_players, score_range=(0, 1))

    def _score(self, X: NDArray) -> float:
        return 1 if len(X) > self.n_players // 2 else 0

    def shapley_values(self) -> ValuationResult:
        exact_values = np.ones(self.n_players) / self.n_players
        result = ValuationResult(
            algorithm="exact_shapley",
            status=Status.Converged,
            indices=self.data.indices,
            values=exact_values,
            variances=np.zeros_like(self.data.data().x),
            counts=np.zeros_like(self.data.data().x),
        )
        return result

    def banzhaf_values(self):
        # There are n-1 choose n/2 coalitions of size n/2, which are the only
        # ones for which the marginal utility is 1.
        exact = math.comb(self.n_players - 1, self.n_players // 2) / 2 ** (
            self.n_players - 1
        )
        result = ValuationResult(
            algorithm="exact_banzhaf",
            status=Status.Converged,
            indices=self.data.indices,
            values=np.full(self.n_players, exact),
            variances=np.zeros_like(self.data.data().x),
            counts=np.zeros_like(self.data.data().x),
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

    def __init__(self, n_players: int = 51):
        if n_players != 51:
            raise ValueError(
                f"{self.__class__.__name__} only supports n_players=51 but got {n_players=}."
            )
        super().__init__(n_players, score_range=(0, 1))

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

    def _score(self, X: NDArray) -> float:
        return 1 if np.sum(self.weight_table[X]) > self.threshold else 0

    @lru_cache
    def shapley_values(self) -> ValuationResult:
        result = ValuationResult(
            algorithm="exact_shapley",
            status=Status.Converged,
            indices=self.data.indices,
            values=self.exact_values,
            variances=np.zeros_like(self.data.data().x),
            counts=np.zeros_like(self.data.data().x),
        )
        return result


class ShoesGame(Game):
    r"""Toy game that is used for testing and demonstration purposes.

    A shoes game defined in (Castro et al.,
    2009)<sup><a href="#castro_polynomial_2009">1</a></sup>.

    In this game, some players have a left shoe and others a right shoe.

    The payoff (utility) of a coalition $S$ is:

    $${
    U(S) = \min( \mid S \cap L \mid, \mid S \cap R \mid )
    }$$

    Where $L$, respectively $R$, is the set of players with left shoes, respectively
    right shoes. This means that the marginal contribution of a player with a left shoe
    to a coalition $S$ is 1 if the number of players with a left shoe in $S$ is strictly
    less than the number of players with a right shoe in $S$, and 0 otherwise. Let
    player $i$ have a left shoe, then:

    $${
    U(S_{+i}) - U(S) =
        \left\{
            \begin{array}{ll}
                1, & \text{ if} \mid S \cap L \mid < \mid S \cap R \mid \\
                0, & \text{ otherwise}
            \end{array}
        \right.
    }$$

    The situation is analogous for players with a right shoe. In order to compute the
    Shapley or Banzhaf value for a player $i$ with a left shoe, we need then the number
    of subsets $S$ of $D_{-i}$ such that $\mid S \cap L \mid < \mid S \cap R \mid$. This
    number is given by the sum:

    $$\sum^{| L |}_{i = 0} \sum_{j > i}^{| R |} \binom{| L |}{i} \binom{| R |}{j}.$$

    Args:
        left: Number of players with a left shoe.
        right: Number of players with a right shoe.
    """

    def __init__(self, left: int, right: int):
        self.left = left
        self.right = right
        n_players = self.left + self.right
        max_score = n_players // 2
        super().__init__(n_players, score_range=(0, max_score))

    def _score(self, X: NDArray) -> float:
        left_sum = float(np.sum(np.asarray(X) < self.left))
        right_sum = float(np.sum(np.asarray(X) >= self.left))
        return min(left_sum, right_sum)

    @lru_cache
    def shapley_values(self) -> ValuationResult:
        """
        We use the fact that the marginal utility of a coalition S of size k is 1 if
        |S ∩ L| < |S ∩ R| and 0 otherwise, and compute Shapley values with the formula
        that iterates over subset sizes.

        The solution for left or right shoes is symmetrical
        """
        left_value = 0.0
        right_value = 0.0
        m = self.n_players - 1
        for k in range(m + 1):
            left_value += (
                1 / math.comb(m, k) * self.n_subsets_left(self.left - 1, self.right, k)
            )
            right_value += (
                1 / math.comb(m, k) * self.n_subsets_right(self.left, self.right - 1, k)
            )
        left_value /= self.n_players
        right_value /= self.n_players
        exact_values = np.array([left_value] * self.left + [right_value] * self.right)
        return ValuationResult(
            algorithm="exact_shapley",
            status=Status.Converged,
            indices=self.data.indices,
            values=exact_values,
            variances=np.zeros_like(self.data.data().x),
            counts=np.zeros_like(self.data.data().x),
        )

    @lru_cache
    def banzhaf_values(self) -> ValuationResult:
        """
        We use the fact that the marginal utility of a coalition S is 1 if
        |S ∩ L| < |S ∩ R| and 0 otherwise, and simply count those sets.

        The solution for left or right shoes is symmetrical.
        """
        m = self.n_players - 1
        left_value = self.n_subsets_left(self.left - 1, self.right) / 2**m
        right_value = self.n_subsets_right(self.left, self.right - 1) / 2**m

        exact_values = np.array([left_value] * self.left + [right_value] * self.right)
        return ValuationResult(
            algorithm="exact_banzhaf",
            status=Status.Converged,
            indices=self.data.indices,
            values=exact_values,
            variances=np.zeros_like(self.data.data().x),
            counts=np.zeros_like(self.data.data().x),
        )

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

        result = ValuationResult(
            algorithm="exact_least_core",
            status=Status.Converged,
            indices=self.data.indices,
            values=exact_values,
            subsidy=subsidy,
            variances=np.zeros_like(self.data.data().x),
            counts=np.zeros_like(self.data.data().x),
        )
        return result

    def least_core_problem(self) -> LeastCoreProblem:
        a_lb = _exact_a_lb(self.left + self.right)
        if self.left == 1 and self.right == 1:
            utilities = np.array([0, 0, 0, 1])
        elif self.left == 2 and self.right == 1:
            utilities = np.array([0] * 5 + [1] * 3)
        elif self.left == 1 and self.right == 2:
            utilities = np.array([0, 0, 0, 0, 1, 1, 0, 1])
        else:
            raise ValueError(
                f"Unsupported game with left={self.left} and right={self.right}"
            )

        return LeastCoreProblem(
            A_lb=a_lb.astype(float), utility_values=utilities.astype(float)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(left={self.left}, right={self.right})"

    @staticmethod
    @lru_cache
    def n_subsets_left(n_left: int, n_right: int, size: int | None = None) -> int:
        acc = 0
        for i in range(n_left + 1):
            for j in range(i + 1, n_right + 1):
                if size is None or i + j == size:
                    acc += math.comb(n_left, i) * math.comb(n_right, j)
        return acc

    @staticmethod
    def n_subsets_right(n_left: int, n_right: int, size: int | None = None) -> int:
        return ShoesGame.n_subsets_left(n_right, n_left, size)


class AirportGame(Game):
    """Toy game that is used for testing and demonstration purposes.

    An airport game defined in
    (Castro et al., 2009)<sup><a href="#castro_polynomial_2009">1</a></sup>
    Section 4.3

    Args:
        n_players: Number of players that participate in the game.
    """

    def __init__(self, n_players: int = 100):
        if n_players != 100:
            raise ValueError(
                f"{self.__class__.__name__} only supports n_players=100 but got {n_players=}."
            )
        super().__init__(n_players, score_range=(0, 100))
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

    def _score(self, X: NDArray) -> float:
        return max(self.score_table[X]) or 0.0

    @lru_cache
    def shapley_values(self) -> ValuationResult:
        result = ValuationResult(
            algorithm="exact_shapley",
            status=Status.Converged,
            indices=self.data.indices,
            values=self.exact_values,
            variances=np.zeros_like(self.data.data().x),
            counts=np.zeros_like(self.data.data().x),
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
        super().__init__(n_players, score_range=(0, np.inf))

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

    def _score(self, X: NDArray) -> float:
        partial_graph = sp.sparse.csr_array(self.graph[np.ix_(X, X)])
        span_tree = sp.sparse.csgraph.minimum_spanning_tree(partial_graph)
        return span_tree.sum() or 0

    @lru_cache
    def shapley_values(self) -> ValuationResult:
        exact_values = 2.0 * np.ones_like(self.data.data().x)
        result = ValuationResult(
            algorithm="exact_shapley",
            status=Status.Converged,
            indices=self.data.indices,
            values=exact_values,
            variances=np.zeros_like(self.data.data().x),
            counts=np.zeros_like(self.data.data().x),
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
        data_description:
    """

    def __init__(self, n_players: int):
        if n_players <= 2:
            raise ValueError(f"n_players, {n_players}, should be > 2")
        super().__init__(n_players, score_range=(0, n_players // 2))

    def _score(self, X: NDArray) -> float:
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

        result = ValuationResult(
            algorithm="exact_least_core",
            status=Status.Converged,
            indices=self.data.indices,
            values=values,
            subsidy=subsidy,
            variances=np.zeros_like(self.data.data().x),
            counts=np.zeros_like(self.data.data().x),
        )
        return result

    def least_core_problem(self) -> LeastCoreProblem:
        a_lb = _exact_a_lb(self.n_players)
        if self.n_players == 3:
            utilities = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        elif self.n_players == 4:
            utilities = np.array([0] * 5 + [1] * 10 + [2])
        else:
            raise NotImplementedError(
                f"Least core problem not implemented for {self.n_players=}"
            )

        return LeastCoreProblem(utility_values=utilities.astype(float), A_lb=a_lb)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={self.n_players})"


def _exact_a_lb(n_players):
    """Hardcoded exact A_lb matrix for testing least-core problem generation."""
    if n_players == 2:
        a_lb = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    elif n_players == 3:
        a_lb = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ]
        )
    elif n_players == 4:
        a_lb = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 0, 1],
                [1, 0, 1, 1],
                [0, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        )
    else:
        raise NotImplementedError(
            "Exact A_lb matrix is not implemented for more than 4 players."
        )
    return a_lb.astype(float)

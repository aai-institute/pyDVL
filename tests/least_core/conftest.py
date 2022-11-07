from typing import Iterable, Tuple

import numpy as np
import pytest

from pydvl.utils import Dataset, Utility


class MinerUtility(Utility):
    """
    Consider a group of n miners, who have discovered large bars of gold.

    If two miners can carry one piece of gold, then the payoff of a coalition S is:

    .. math::

        v(S) = \left\{\begin{matrix}
            |S|/2 & \text{, if} & |S| \text{ is even} \\
            (|S| - 1)/2 & \text{, if} & |S| \text{ is odd}
        \end{matrix}\right.

    If there are more than two miners and there is an even number of miners,
    then the core consists of the single payoff where each miner gets 1/2.

    If there is an odd number of miners, then the core is empty.

    Taken from: https://en.wikipedia.org/wiki/Core_(game_theory)
    """

    def __init__(self, n_miners: int):
        if n_miners <= 2:
            raise ValueError(f"n_miners, {n_miners} should be > 2")
        self.n_miners = n_miners

        x = np.arange(n_miners)[..., np.newaxis]
        # The y values don't matter here
        y = np.zeros_like(x)

        self.data = Dataset(x_train=x, y_train=y, x_test=x, y_test=y)

    def __call__(self, indices: Iterable[int]) -> float:
        n = len(tuple(indices))
        if n % 2 == 0:
            return n / 2
        else:
            return (n - 1) / 2

    def _initialize_utility_wrapper(self):
        pass


@pytest.fixture()
def miner_utility(request) -> Tuple[Utility, dict]:
    n_miners = request.param
    u = MinerUtility(n_miners=n_miners)
    if n_miners % 2 == 0:
        exact_values = {i: 0.5 for i in range(n_miners)}
    else:
        exact_values = {i: (n_miners - 1) / (2 * n_miners) for i in range(n_miners)}
    return u, exact_values

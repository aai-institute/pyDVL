import numpy as np

from pydvl.valuation import Dataset
from pydvl.valuation.methods.random import RandomValuation


def test_random_reproducible(seed):
    valuation = RandomValuation(seed)
    train, _ = Dataset.from_arrays(np.random.rand(100, 10), np.random.rand(100))

    valuation.fit(train)
    result1 = valuation.result

    valuation = RandomValuation(seed)
    valuation.fit(train)
    result2 = valuation.result

    assert result1 == result2


def test_random_fit_different(seed):
    valuation = RandomValuation(seed)
    train, _ = Dataset.from_arrays(np.random.rand(100, 10), np.random.rand(100))

    valuation.fit(train)
    result1 = valuation.result

    valuation.fit(train)
    result2 = valuation.result

    assert result1 != result2

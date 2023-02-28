import numpy as np
import pytest

from pydvl.utils.status import Status
from pydvl.value import ValuationResult


@pytest.fixture(scope="function")
def analytic_loo(dummy_utility):
    r"""Scores are i/m, so v(i) = U(D) - U(D\{i})] = i/m"""

    m = float(max(dummy_utility.data.x_train))
    values = np.array([i / m for i in dummy_utility.data.indices])

    result = ValuationResult(
        algorithm="exact",
        values=values,
        variances=np.zeros_like(values),
        data_names=dummy_utility.data.indices,
        status=Status.Converged,
    )
    return dummy_utility, result

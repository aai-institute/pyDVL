import numpy as np
import pytest

from pydvl.value import ValuationResult, ValuationStatus


@pytest.fixture(scope="function")
def analytic_loo(dummy_utility):
    """Scores are i/m, so v(i) = U(D) - U(D\{i})] = i/m"""

    m = float(max(dummy_utility.data.x_train))
    values = np.array([i / m for i in dummy_utility.data.indices])

    result = ValuationResult(
        algorithm="exact",
        values=values,
        steps=1,
        stderr=np.zeros_like(values),
        data_names=dummy_utility.data.indices,
        status=ValuationStatus.Converged,
    )
    return dummy_utility, result

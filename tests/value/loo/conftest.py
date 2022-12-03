import numpy as np
import pytest

from pydvl.utils import SortOrder
from pydvl.value import ValuationResult, ValuationStatus


@pytest.fixture(scope="function")
def analytic_loo(dummy_utility):
    """Scores are i/m, so v(i) = U(D) - U(D\{i})] = i/m"""

    def exact():
        pass

    m = float(max(dummy_utility.data.x_train))
    values = np.array([i / m for i in dummy_utility.data.indices])

    result = ValuationResult(
        algorithm=exact,
        values=values,
        stderr=np.zeros_like(values),
        data_names=dummy_utility.data.indices,
        sort=SortOrder.Descending,
        status=ValuationStatus.Converged,
    )
    return dummy_utility, result

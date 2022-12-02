import numpy as np
import pytest


@pytest.fixture(scope="session")
def analytic_loo(num_samples, dummy_utility):
    """Scores are i/m, so v(i) = U(D) - U(D\{i})] = (m-i)/m"""

    u = dummy_utility(num_samples)
    m = float(max(u.data.x_train))
    exact_values = np.array([(m - i) / m for i in u.data.indices])
    return u, exact_values

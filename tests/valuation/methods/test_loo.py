import joblib
import numpy as np
import pytest

from pydvl.utils.status import Status
from pydvl.valuation import LOOValuation
from pydvl.valuation.result import ValuationResult

from .. import check_total_value, check_values


@pytest.fixture(scope="function")
def analytic_loo(dummy_train_data):
    r"""Scores are i/m, so v(i) = U(D) - U(D\{i})] = i/m"""
    x, _ = dummy_train_data.data()
    m = float(max(x))
    values = np.array([i / m for i in dummy_train_data.indices])
    result = ValuationResult(
        algorithm="exact",
        values=values,
        variances=np.zeros_like(values),
        data_names=dummy_train_data.indices,
        status=Status.Converged,
    )
    return result


# num_samples indirectly parametrizes the fixtures
@pytest.mark.parametrize("num_samples", [10, 100])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_loo(dummy_utility, dummy_train_data, analytic_loo, n_jobs):
    """Compares LOO with analytic values in a dummy model"""
    valuation = LOOValuation(utility=dummy_utility, progress=False)
    with joblib.parallel_config(backend="loky", n_jobs=n_jobs):
        valuation.fit(dummy_train_data)
    got = valuation.values()
    check_total_value(dummy_utility, got, rtol=0.1)
    check_values(got, analytic_loo, rtol=0.1)


@pytest.mark.parametrize("n", [1, 100])
def test_loo_coefficients_cancel(n):
    """Check that the coefficients sum to 1"""
    valuation = LOOValuation(utility=None, progress=False)  # type: ignore
    sampler = valuation.sampler
    corrections = [
        valuation.coefficient(n, k, sampler.weight(n, k)) for k in range(n + 1)
    ]
    np.testing.assert_allclose(sum(corrections), 1.0, atol=1e-10)


@pytest.mark.parametrize("n", [1, 100])
def test_loo_log_coefficients_cancel(n):
    """Check that the coefficients sum to 1"""
    valuation = LOOValuation(utility=None, progress=False)  # type: ignore
    sampler = valuation.sampler
    corrections = [
        valuation.log_coefficient(n, k) + sampler.log_weight(n, k) for k in range(n + 1)
    ]
    np.testing.assert_allclose(np.exp(corrections).sum(), 1.0, atol=1e-10)

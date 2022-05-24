import numpy as np
import pytest
import logging

from tests.conftest import check_exact, check_total_value
from valuation.shapley import combinatorial_exact_shapley, permutation_exact_shapley
from sklearn.linear_model import LinearRegression
from valuation.utils import Utility


# noinspection PyTestParametrized
@pytest.mark.parametrize(
        "num_samples, fun, value_atol, total_atol",
        [(12, combinatorial_exact_shapley, 1e-5, 1e-5),
         (6, permutation_exact_shapley, 1e-5, 1e-5)])
def test_analytic_exact_shapley(analytic_shapley, fun, value_atol, total_atol):
    """Compares the combinatorial exact shapley and permutation exact shapley with
    the analytic_shapley calculation for a dummy model.
    """
    u, exact_values = analytic_shapley
    values_p = fun(u, progress=False)
    check_total_value(u, values_p, atol=total_atol)
    check_exact(values_p, exact_values, atol=value_atol)

@pytest.mark.parametrize(
        "a, b, score_type",
        [(2, 0, "r2"), (2, 1 ,"r2"), (2, 1, "neg_median_absolute_error"),  (2, 1, "explained_variance")]
        )
def test_linear(linear_dataset, score_type, value_atol=1e-5, total_atol=1e-5):
    linear_utility = Utility(LinearRegression(), data=linear_dataset, scoring=score_type)
    values_combinatorial = combinatorial_exact_shapley(linear_utility, progress=False)
    check_total_value(linear_utility, values_combinatorial, atol=total_atol)

    values_permutation = permutation_exact_shapley(linear_utility, progress=False)
    check_total_value(linear_utility, values_permutation, atol=total_atol)

    check_exact(values_combinatorial, values_permutation, atol=value_atol)

@pytest.mark.parametrize(
        "a, b, score_type",
        [(2, 0, "r2"), (2, 1,"r2"), (2, 1, "neg_median_absolute_error"),  (2, 1, "explained_variance")]
        )
def test_linear_with_outlier(linear_dataset, score_type, total_atol=1e-5):
    outlier_idx = np.random.randint(len(linear_dataset.y_train))
    linear_dataset.y_train[outlier_idx] *= 10
    linear_utility = Utility(LinearRegression(), data=linear_dataset, scoring=score_type)
    shapley_values = permutation_exact_shapley(linear_utility, progress=False)
    logging.info(f'shapley values: {shapley_values}')
    check_total_value(linear_utility, shapley_values, atol=total_atol)

    assert int(list(shapley_values.keys())[0]) == outlier_idx

@pytest.mark.parametrize(
        "coefficients, score_type",
        [(np.random.randint(-3, 3, size=3), "r2"),
        (np.random.randint(-3, 3, size=5), "neg_median_absolute_error"), 
        (np.random.randint(-3, 3, size=7), "explained_variance")]
        )
def test_polynomial(polynomial_dataset, polynomial_pipeline, score_type, value_atol=1e-5, total_atol=1e-5):
    dataset, _ = polynomial_dataset
    poly_utility = Utility(polynomial_pipeline, dataset, scoring=score_type)
    values_combinatorial = combinatorial_exact_shapley(poly_utility, progress=False)
    check_total_value(poly_utility, values_combinatorial, atol=total_atol)

    values_permutation = permutation_exact_shapley(poly_utility, progress=False)
    check_total_value(poly_utility, values_permutation, atol=total_atol)

    check_exact(values_combinatorial, values_permutation, atol=value_atol)

@pytest.mark.parametrize(
        "coefficients, score_type",
        [(np.random.randint(-3, 3, size=3), "r2"),
        (np.random.randint(-3, 3, size=5), "neg_median_absolute_error"), 
        (np.random.randint(-3, 3, size=7), "explained_variance")]
        )
def test_polynomial_with_outlier(polynomial_dataset, polynomial_pipeline, score_type, total_atol=1e-5):
    dataset, _ = polynomial_dataset
    outlier_idx = np.random.randint(len(dataset.y_train))
    dataset.y_train[outlier_idx] *= 10
    poly_utility = Utility(polynomial_pipeline, dataset, scoring=score_type)
    shapley_values = permutation_exact_shapley(poly_utility, progress=False)
    logging.info(f'shapley values: {shapley_values}')
    check_total_value(poly_utility, shapley_values, atol=total_atol)

    assert int(list(shapley_values.keys())[0]) == outlier_idx




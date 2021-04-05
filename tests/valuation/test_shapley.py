import numpy as np
from sklearn.linear_model import LinearRegression
from valuation.shapley.naive import exact_combinatorial_shapley, \
    exact_permutation_shapley


def test_exact_naive_shapley(linear_dataset):
    reg = LinearRegression()
    values_p = exact_permutation_shapley(reg, linear_dataset, progress=False)
    values_c = exact_combinatorial_shapley(reg, linear_dataset, progress=False)

    assert np.alltrue(values_p.keys() == values_c.keys())
    assert np.allclose(values_p.values(), values_c.values())

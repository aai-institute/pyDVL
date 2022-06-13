import pytest

from tests.conftest import check_exact, check_total_value
from valuation.shapley import (
    combinatorial_exact_shapley,
    combinatorial_montecarlo_shapley,
    permutation_exact_shapley,
    permutation_montecarlo_shapley,
    truncated_montecarlo_shapley,
)

pytestmark = [
    pytest.mark.benchmark(
        group="shapley",
        min_rounds=5,
    ),
    pytest.mark.parametrize("num_samples", [3, 6]),
]


@pytest.mark.parametrize(
    "method",
    [
        combinatorial_exact_shapley,
        permutation_exact_shapley,
    ],
)
def test_exact_shapley_performance(exact_shapley, method, benchmark):
    u, exact_values = exact_shapley
    benchmark(method, u, progress=False)


@pytest.mark.parametrize("max_iterations", [10, 20])
@pytest.mark.parametrize(
    "method",
    [
        combinatorial_montecarlo_shapley,
        permutation_montecarlo_shapley,
    ],
)
def test_montecarlo_shapley_performance(
    exact_shapley, method, max_iterations, benchmark
):
    u, exact_values = exact_shapley
    benchmark(method, u, max_iterations=max_iterations, progress=False, use_cache=False)


@pytest.mark.skip
@pytest.mark.parametrize("max_iterations", [10, 20])
def test_truncated_montecarlo_shapley_performance(
    exact_shapley, max_iterations, benchmark
):
    u, exact_values = exact_shapley
    benchmark(
        truncated_montecarlo_shapley, u, max_iterations=max_iterations, progress=False
    )

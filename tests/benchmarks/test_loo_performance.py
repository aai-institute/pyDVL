import pytest

from tests.conftest import check_exact, check_total_value
from valuation.loo.naive import naive_loo

pytestmark = [
    pytest.mark.benchmark(
        group="loo",
        min_rounds=5,
    ),
    pytest.mark.parametrize("num_samples", [3, 5, 10]),
    pytest.mark.timeout(3),
]


@pytest.mark.parametrize(
    "method",
    [
        naive_loo,
    ],
)
def test_loo(exact_loo, method, benchmark):
    u, exact_values = exact_loo
    values = benchmark(
        method, data=u.data, model=u.model, progress=False, enable_cache=False
    )
    check_exact(values, exact_values)

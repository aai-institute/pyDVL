import numpy as np
import pytest

from pydvl.valuation.methods import DeltaShapleyValuation


@pytest.mark.parametrize("n", [10, 100])
def test_log_coefficient(n):
    valuation = DeltaShapleyValuation(  # type: ignore
        utility=None,
        is_done=None,
        lower_bound=n // 3,
        upper_bound=2 * n // 3,
        progress=False,
    )

    s = [valuation.log_coefficient(n, j) for j in range(n + 1)]
    np.testing.assert_allclose(1, np.sum(np.exp(s)))

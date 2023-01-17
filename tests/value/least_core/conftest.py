from typing import Tuple

import numpy as np
import pytest

from pydvl.utils import Utility
from pydvl.utils.utility import MinerUtility
from pydvl.value.results import ValuationResult, ValuationStatus


@pytest.fixture()
def miner_utility(request) -> Tuple[Utility, ValuationResult]:
    n_miners = request.param
    u = MinerUtility(n_miners=n_miners)
    exact_values, subsidy = u.exact_least_core_values()
    result = ValuationResult(
        algorithm="exact",
        values=exact_values,
        subsidy=subsidy,
        stderr=np.zeros_like(exact_values),
        data_names=np.arange(len(exact_values)),
        status=ValuationStatus.Converged,
    )
    return u, result

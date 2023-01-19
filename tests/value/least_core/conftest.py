from typing import Tuple

import numpy as np
import pytest

from pydvl.utils import Utility
from pydvl.utils.status import Status
from pydvl.utils.utility import GlovesGameUtility, MinerGameUtility
from pydvl.value.results import ValuationResult


@pytest.fixture(scope="module")
def test_utility(request) -> Tuple[Utility, ValuationResult]:
    name, kwargs = request.param
    if name == "miner":
        u = MinerGameUtility(**kwargs)
    elif name == "gloves":
        u = GlovesGameUtility(**kwargs)
    else:
        raise ValueError(f"Unknown '{name}'")
    exact_values, subsidy = u.exact_least_core_values()
    result = ValuationResult(
        algorithm="exact",
        values=exact_values,
        subsidy=subsidy,
        stderr=np.zeros_like(exact_values),
        data_names=np.arange(len(exact_values)),
        status=Status.Converged,
    )
    return u, result

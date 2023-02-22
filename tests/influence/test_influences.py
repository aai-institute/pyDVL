import itertools
from typing import List, Tuple

import numpy as np
import pytest

from .conftest import (
    analytical_linear_influences,
    create_mock_dataset,
    linear_analytical_influence_factors,
    linear_model,
)

try:
    import torch
    import torch.nn.functional as F
    from torch import nn

    from pydvl.influence.frameworks import TorchTwiceDifferentiable
    from pydvl.influence.general import (
        InfluenceType,
        calculate_influence_factors,
        compute_influences,
    )
except ImportError:
    pass


class InfluenceTestSettings:
    ACCEPTABLE_RTOL_INFLUENCE: float = 1e-7
    ACCEPTABLE_RTOL_INFLUENCE_CG: float = 1e-1
    INFLUENCE_TYPE: List[InfluenceType] = [InfluenceType.Up, InfluenceType.Perturbation]
    INFLUENCE_TEST_CONDITION_NUMBERS: List[int] = [3]
    INFLUENCE_TRAINING_SET_SIZE: List[int] = [50, 30]
    INFLUENCE_TEST_SET_SIZE: List[int] = [20]
    INFLUENCE_DIMENSIONS: List[Tuple[int, int]] = [
        (10, 10),
        (3, 20),
    ]
    HESSIAN_REGULARIZATION: List[float] = [0, 1]


test_cases = list(
    itertools.product(
        InfluenceTestSettings.INFLUENCE_TRAINING_SET_SIZE,
        InfluenceTestSettings.INFLUENCE_TEST_SET_SIZE,
        InfluenceTestSettings.INFLUENCE_TYPE,
        InfluenceTestSettings.INFLUENCE_DIMENSIONS,
        InfluenceTestSettings.INFLUENCE_TEST_CONDITION_NUMBERS,
        InfluenceTestSettings.HESSIAN_REGULARIZATION,
    )
)


def lmb_test_case_to_str(packed_i_test_case):
    i, test_case = packed_i_test_case
    return (
        f"Problem #{i} of dimension {test_case[3]} with train size {test_case[0]}, "
        f"test size {test_case[1]}, if_type {test_case[2]}, condition number {test_case[4]} and lam {test_case[5]}."
    )


test_case_ids = list(map(lmb_test_case_to_str, zip(range(len(test_cases)), test_cases)))


@pytest.mark.torch
@pytest.mark.parametrize(
    "train_set_size,test_set_size,influence_type,problem_dimension,condition_number, hessian_reg",
    test_cases,
    ids=test_case_ids,
)
def test_linear_influence(
    train_set_size: int,
    test_set_size: int,
    influence_type: InfluenceType,
    problem_dimension: Tuple[int, int],
    condition_number: float,
    hessian_reg: float,
):
    A, b = linear_model(problem_dimension, condition_number)
    train_data, test_data = create_mock_dataset((A, b), train_set_size, test_set_size)

    linear_layer = nn.Linear(A.shape[0], A.shape[1])
    linear_layer.weight.data = torch.as_tensor(A)
    linear_layer.bias.data = torch.as_tensor(b)
    loss = F.mse_loss

    analytical_influences = analytical_linear_influences(
        (A, b),
        *train_data,
        *test_data,
        influence_type=influence_type,
        hessian_regularization=hessian_reg,
    )

    direct_influences = compute_influences(
        linear_layer,
        loss,
        *train_data,
        *test_data,
        progress=True,
        influence_type=influence_type,
        inversion_method="direct",
        hessian_regularization=hessian_reg,
    )

    cg_influences = compute_influences(
        linear_layer,
        loss,
        *train_data,
        *test_data,
        progress=True,
        influence_type=influence_type,
        inversion_method="cg",
        hessian_regularization=hessian_reg,
    )
    assert np.logical_not(np.any(np.isnan(direct_influences)))
    assert np.logical_not(np.any(np.isnan(cg_influences)))
    direct_if_max_abs_diff = np.max(
        np.abs((direct_influences - analytical_influences) / analytical_influences)
    )
    assert (
        direct_if_max_abs_diff < InfluenceTestSettings.ACCEPTABLE_RTOL_INFLUENCE
    ), f"Influence values of type {influence_type} for direct inversion are wrong: max relative diff {direct_if_max_abs_diff}"
    cg_if_max_abs_diff = np.max(
        np.abs((cg_influences - analytical_influences) / analytical_influences)
    )
    assert (
        cg_if_max_abs_diff < InfluenceTestSettings.ACCEPTABLE_RTOL_INFLUENCE_CG
    ), f"Influence values of type {influence_type} for cg inversion are wrong: max relative diff {cg_if_max_abs_diff}"

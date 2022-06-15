import itertools
from typing import List

import numpy as np
import pytest
import torch.nn.functional as F

from valuation.models.linear_regression_torch_model import LRTorchModel
from valuation.models.pytorch_model import PyTorchOptimizer, PyTorchSupervisedModel


class ModelTestSettings:
    DATA_OUTPUT_NOISE: float = 0.01
    ACCEPTABLE_ABS_TOL_MODEL: float = 0.02

    TEST_CONDITION_NUMBERS: List[int] = [5]
    TEST_SET_SIZE: List[int] = [10, 20]
    TRAINING_SET_SIZE: List[int] = [500, 1000]
    DIMENSIONS: List[int] = [2, 5, 10]


test_cases = list(
    itertools.product(
        ModelTestSettings.TRAINING_SET_SIZE,
        ModelTestSettings.TEST_SET_SIZE,
        ModelTestSettings.DIMENSIONS,
        ModelTestSettings.TEST_CONDITION_NUMBERS,
    )
)


def lmb_test_case_to_str(packed_i_test_case):
    i, test_case = packed_i_test_case
    return f"Problem #{i} of dimension {test_case[2]} with train size {test_case[0]}, test size {test_case[1]} and condition number {test_case[3]}"


test_case_ids = list(map(lmb_test_case_to_str, zip(range(len(test_cases)), test_cases)))


@pytest.mark.parametrize(
    "train_set_size,test_set_size,problem_dimension,condition_number",
    test_cases,
    ids=test_case_ids,
)
def test_linear_regression_model(
    train_set_size: int,
    test_set_size: int,
    problem_dimension: int,
    condition_number: float,
    quadratic_matrix: np.ndarray,
):

    # some settings
    A = quadratic_matrix
    d, _ = tuple(A.shape)

    # generate datasets
    data_model = lambda x: np.random.normal(
        x @ A.T, ModelTestSettings.DATA_OUTPUT_NOISE
    )
    train_x = np.random.uniform(size=[train_set_size, d])
    train_y = data_model(train_x)
    test_x = np.random.uniform(size=[test_set_size, d])
    test_y = data_model(test_x)

    model = PyTorchSupervisedModel(
        model=LRTorchModel(d, d),
        objective=F.mse_loss,
        num_epochs=1000,
        batch_size=32,
        optimizer=PyTorchOptimizer.ADAM_W,
        optimizer_kwargs={"lr": 0.05},
    )
    model.fit(train_x, train_y)
    learned_A = model.model.A.detach().numpy()
    max_A_diff = np.max(np.abs(learned_A - A))
    assert (
        max_A_diff < ModelTestSettings.ACCEPTABLE_ABS_TOL_MODEL
    ), "Model did not converged to target solution."

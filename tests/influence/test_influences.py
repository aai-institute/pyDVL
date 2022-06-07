import itertools
from functools import partial

import numpy as np
import pytest
import torch.nn.functional as F

from valuation.influence.naive import influences
from valuation.models.linear_regression_torch_model import LRTorchModel
from valuation.models.neural_network_torch_model import NNTorchModel
from valuation.models.pytorch_model import PyTorchSupervisedModel
from valuation.utils import Dataset


@pytest.mark.parametrize(
    "torch_model_factory,n_jobs",
    list(
        itertools.product(
            [
                partial(NNTorchModel, n_neurons_per_Layer=[8, 8]),
                LRTorchModel,
            ],
            [1, 2],
        )
    ),
)
def test_influences(
    linear_dataset: Dataset,
    torch_model_factory,
    n_jobs: int,
):
    n_in_features = linear_dataset.x_test.shape[1]

    model = PyTorchSupervisedModel(
        model=torch_model_factory(n_in_features, 1), objective=F.mse_loss, num_epochs=10
    )
    model.fit(linear_dataset.x_train, linear_dataset.y_train)
    influence_values = influences(model, linear_dataset, progress=True, n_jobs=n_jobs)

    assert np.all(np.logical_not(np.isnan(influence_values)))
    assert influence_values.shape == (
        len(linear_dataset.x_test),
        len(linear_dataset.x_train),
    )

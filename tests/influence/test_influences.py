from collections import OrderedDict
from functools import partial

import pytest
import torch.nn.functional as F
import torch.random

from valuation.influence.naive import influences
from valuation.models.linear_regression_torch_model import LRTorchModel
from valuation.models.neural_network_torch_model import NNTorchModel
from valuation.models.pytorch_model import PyTorchOptimizer, PyTorchSupervisedModel
from valuation.utils import Dataset

test_cases = OrderedDict()
test_cases["nn_test_single_thread"] = (
    partial(NNTorchModel, n_neurons_per_Layer=[8, 8]),
    1,
)
test_cases["nn_test_multi_thread"] = (
    partial(NNTorchModel, n_neurons_per_Layer=[8, 8]),
    2,
)
test_cases["lr_test_single_thread"] = (LRTorchModel, 1)
test_cases["lr_test_multi_thread"] = (LRTorchModel, 2)


@pytest.mark.parametrize(
    "torch_model_factory,n_jobs", test_cases.values(), ids=test_cases.keys()
)
def test_influences(linear_dataset: Dataset, torch_model_factory, n_jobs: int):
    n_in_features = linear_dataset.x_test.shape[1]
    print(torch.random.seed())
    model = PyTorchSupervisedModel(
        model=torch_model_factory(n_in_features, 1),
        objective=F.mse_loss,
        num_epochs=10,
        batch_size=16,
        optimizer=PyTorchOptimizer.ADAM,
        optimizer_kwargs={"lr": 0.05},
    )
    model.fit(linear_dataset.x_train, linear_dataset.y_train)
    influence_values = influences(model, linear_dataset, progress=True, n_jobs=n_jobs)

    # assert np.all(np.logical_not(np.isnan(influence_values)))
    assert influence_values.shape == (
        len(linear_dataset.x_test),
        len(linear_dataset.x_train),
    )

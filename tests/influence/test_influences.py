from collections import OrderedDict

import numpy as np
import pytest
import torch.nn.functional as F

from valuation.influence.naive import influences
from valuation.models.linear_regression_torch_model import LRTorchModel
from valuation.models.pytorch_model import PyTorchOptimizer, PyTorchSupervisedModel
from valuation.utils import Dataset

test_cases = OrderedDict()
test_cases["lr_test_single_thread"] = (LRTorchModel, 1)
test_cases["lr_test_multi_thread"] = (LRTorchModel, 2)


@pytest.mark.parametrize(
    "torch_model_factory,n_jobs", test_cases.values(), ids=test_cases.keys()
)
def test_influences(linear_dataset: Dataset, torch_model_factory, n_jobs: int):
    n_in_features = linear_dataset.x_test.shape[1]
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

    assert np.all(np.logical_not(np.isnan(influence_values)))
    assert influence_values.shape == (
        len(linear_dataset.x_test),
        len(linear_dataset.x_train),
    )


def test_influences_lr_analytical():

    L = np.asarray([[1, 2, 3], [1, 2, 0], [0, 1, 1]])
    A = L @ L.T
    o_d, i_d = tuple(A.shape)
    train_x = np.random.uniform(size=[1000, i_d])
    train_y = train_x @ A.T
    model = PyTorchSupervisedModel(
        model=LRTorchModel(i_d, o_d),
        objective=F.mse_loss,
        num_epochs=1000,
        batch_size=32,
        optimizer=PyTorchOptimizer.ADAM_W,
        optimizer_kwargs={"lr": 0.01},
    )
    model.fit(train_x, train_y)
    learned_A = model.model.A.detach().numpy()
    max_A_diff = np.max(np.abs(learned_A - A))
    assert max_A_diff < 1e-2

    H = A.T @ A
    inv_H = np.linalg.pinv(H)
    s_test = train_x @ inv_H.T
    test_x = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    test_y = test_x @ A.T
    real_influences = -np.einsum("ci,di->cd", test_x, s_test)

    class Object(object):
        pass

    dataset = Object()
    dataset.x_train = train_x
    dataset.y_train = train_y
    dataset.x_test = test_x
    dataset.y_test = test_y
    influence_values = influences(model, dataset, progress=True, n_jobs=1)
    max_val = np.max(np.abs(influence_values - real_influences))
    assert max_val < 1e-2

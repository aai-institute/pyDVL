import numpy as np
import pytest
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from valuation.influence_functions.main import influence_functions
from valuation.models.linear_regression_torch_model import LRTorchModel
from valuation.models.pytorch_model import PyTorchSupervisedModel
from valuation.models.twice_differentiable import TwiceDifferentiable
from valuation.utils import Utility, Dataset


def create_random_dataset(
    n_in_features: int = 3,
    n_out_features: int = 1,
    n_samples: int = 1000
):
    x = np.random.normal(0, 0.1, size=[n_samples, n_in_features])
    A = np.random.uniform(-1, 1.0, size=[n_out_features, n_in_features])
    y = x @ A.T
    y = y.ravel()
    return x, y


@pytest.mark.parametrize(
    "x,y", [create_random_dataset(i, n_samples=1000) for i in range(1, 5)]
)
def test_influences(
    x: np.ndarray,
    y: np.ndarray,
    train_size: int = 0.2
):
    n_in_features = x.shape[1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=train_size)
    dataset = Dataset(x_train, y_train, x_test, y_test)

    objective = F.mse_loss
    model = PyTorchSupervisedModel(LRTorchModel(n_in_features, 1), objective)
    model.fit(dataset.x_train, dataset.y_train)
    utility = Utility(model, dataset, objective)
    batch_influence_functions = influence_functions(utility, progress=True)

    twd: TwiceDifferentiable = model
    test_grads = twd.grad(dataset.x_test, dataset.x_test)
    influences = batch_influence_functions(test_grads)
    assert np.all(np.logical_not(np.isnan(influences)))
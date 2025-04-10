from typing import Tuple

import pytest
import torch
from numpy.typing import NDArray
from torch.optim import LBFGS
from torch.utils.data import DataLoader

from pydvl.influence.torch.util import flatten_dimensions
from tests.influence.conftest import linear_hessian_analytical, linear_model

DATA_OUTPUT_NOISE: float = 0.01


def linear_mvp_model(A, b):
    output_dimension, input_dimension = tuple(A.shape)
    model = torch.nn.Linear(input_dimension, output_dimension)
    model.eval()
    model.weight.data = torch.as_tensor(A)
    model.bias.data = torch.as_tensor(b)
    return model


def minimal_training(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_function: torch.nn.modules.loss._Loss,
    lr: float = 0.01,
    epochs: int = 50,
):
    """
    Trains a PyTorch model using L-BFGS optimizer.

    Args:
        model: The PyTorch model to be trained.
        dataloader: DataLoader providing the training data.
        loss_function: The loss function to be used for training.
        lr: The learning rate for the L-BFGS optimizer. Defaults to 0.01.
        epochs: The number of training epochs. Defaults to 50.

    Returns:
        The trained model.
    """
    model = model.train()
    optimizer = LBFGS(model.parameters(), lr=lr)

    for epoch in range(epochs):
        data = torch.cat([inputs for inputs, targets in dataloader])
        targets = torch.cat([targets for inputs, targets in dataloader])

        def closure():
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_function(outputs, targets)
            loss.backward()
            return loss

        optimizer.step(closure)

    return model


def torch_linear_model_to_numpy(model: torch.nn.Linear) -> Tuple[NDArray, NDArray]:
    model.eval()
    return model.weight.data.numpy(), model.bias.data.numpy()


@pytest.fixture(scope="session")
def device(request):
    import torch

    use_cuda = request.config.getoption("--with-cuda")
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def linear_torch_model_from_numpy(A: NDArray, b: NDArray) -> torch.nn.Module:
    """
    Given numpy arrays representing the model $xA^t + b$, the function returns the corresponding torch model
    :param A:
    :param b:
    :return:
    """
    output_dimension, input_dimension = tuple(A.shape)
    model = torch.nn.Linear(input_dimension, output_dimension)
    model.eval()
    model.weight.data = torch.as_tensor(A, dtype=torch.get_default_dtype())
    model.bias.data = torch.as_tensor(b, dtype=torch.get_default_dtype())
    return model


@pytest.fixture
def model_data(request):
    dimension, condition_number, train_size = request.param
    A, b = linear_model(dimension, condition_number)
    x = torch.rand(train_size, dimension[-1])
    y = torch.rand(train_size, dimension[0])
    torch_model = linear_torch_model_from_numpy(A, b)
    vec = flatten_dimensions(
        tuple(
            torch.rand(*p.shape)
            for name, p in torch_model.named_parameters()
            if p.requires_grad
        )
    )
    H_analytical = linear_hessian_analytical((A, b), x.numpy())
    H_analytical = torch.as_tensor(H_analytical)
    return torch_model, x, y, vec, H_analytical.to(torch.float32)

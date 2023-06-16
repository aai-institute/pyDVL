from dataclasses import dataclass
from typing import Tuple

import pytest
import numpy as np
import scipy
import torch

from torch.utils.data import DataLoader, TensorDataset

from pydvl.influence.frameworks.util import get_hvp_function, lanzcos_low_rank_hessian_approx, hvp
from tests.influence.conftest import linear_model, linear_hessian_analytical, linear_torch_model_from_numpy


@dataclass
class UtilTestParameters:
    """
    Helper class to add more test parameter combinations
    """
    dimension: Tuple[int, int]
    condition_number: float
    train_size: int
    batch_size: int
    rank_estimate: int
    regularization: float


test_parameters = [
    UtilTestParameters(dimension=(30, 16), condition_number=4, train_size=60, batch_size=4, rank_estimate=50,
                       regularization=0.0001),
    UtilTestParameters(dimension=(32, 35), condition_number=1e6, train_size=100, batch_size=5, rank_estimate=70,
                       regularization=0.00001),
    UtilTestParameters(dimension=(25, 15), condition_number=1e3, train_size=90, batch_size=10, rank_estimate=50,
                       regularization=0.000001),
    UtilTestParameters(dimension=(30, 15), condition_number=1e4, train_size=120, batch_size=8, rank_estimate=60,
                       regularization=0.00001),
    UtilTestParameters(dimension=(40, 13), condition_number=1e5, train_size=900, batch_size=4, rank_estimate=90,
                       regularization=0.00001),
]


dimensions = [(30, 16), (32, 35), (25, 15), (30, 15), (40, 13)]
condition_numbers = [4, 1e6, 1e3, 1e4, 1e5]
train_sizes = [60, 100, 90, 120, 900]
batch_sizes = [4, 5, 10, 8, 4]
rank_estimates = [50, 70, 50, 60, 90]
regularizations = [0.0001, 0.00001, 0.000001, 0.00001, 0.00001]


problem_params = [(tp.dimension, tp.condition_number, tp.train_size) for tp in test_parameters]


@pytest.fixture(params=problem_params)
def model_data(request):
    dimension, condition_number, train_size = request.param  # type: ignore
    A, b = linear_model(dimension, condition_number)
    x = np.random.uniform(size=[train_size, dimension[-1]])
    y = np.random.uniform(size=[train_size, dimension[0]])
    torch_model = linear_torch_model_from_numpy(A, b)
    num_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
    vec = np.random.uniform(size=(num_params, ))
    H_analytical = linear_hessian_analytical((A, b), x)
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    vec = torch.as_tensor(vec)
    H_analytical = torch.as_tensor(H_analytical)
    return torch_model, x, y, vec, H_analytical


@pytest.mark.parametrize("model_data, tol",
                         [(problem_param, 1e-12) for problem_param in problem_params],
                         indirect=["model_data"])
def test_hvp(model_data, tol: float):
    torch_model, x, y, vec, H_analytical = model_data
    Hvp_autograd = hvp(torch_model, torch.nn.functional.mse_loss, x, y, vec)
    assert torch.allclose(Hvp_autograd, H_analytical@vec, rtol=tol)


@pytest.mark.parametrize("use_avg, tol", [(True, 1e-3), (False, 1e-6)], ids=["avg", "full"])
@pytest.mark.parametrize("model_data, batch_size",
                         list(zip(problem_params, batch_sizes)),
                         indirect=["model_data"])
def test_get_hvp_function(model_data, tol: float, use_avg: bool, batch_size: int):
    torch_model, x, y, vec, H_analytical = model_data
    data_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
    Hvp_autograd = get_hvp_function(torch_model, torch.nn.functional.mse_loss, data_loader,
                                    use_hessian_avg=use_avg)(vec)
    assert torch.allclose(Hvp_autograd, H_analytical@vec, rtol=tol)


@pytest.mark.parametrize("model_data, batch_size, rank_estimate, regularization",
                         list(zip(problem_params, batch_sizes, rank_estimates, regularizations)),
                         indirect=["model_data"])
def test_lanzcos_low_rank_hessian_approx(model_data, batch_size: int, rank_estimate, regularization):
    _, _, _, vec, H_analytical = model_data

    # artificially make the analytical hessian low rank
    reg_H_analytical = H_analytical + regularization * torch.eye(H_analytical.shape[0])
    eig_val_analytical, proj_analytical = scipy.linalg.eigh(reg_H_analytical.numpy())
    eig_val_analytical, proj_analytical = torch.as_tensor(eig_val_analytical), torch.as_tensor(proj_analytical)
    idx_analytical = torch.argsort(torch.abs(eig_val_analytical), descending=True)
    eig_val_analytical = eig_val_analytical[idx_analytical[:rank_estimate]]
    proj_analytical = proj_analytical[:, idx_analytical[:rank_estimate]]
    low_rank_approx_analytical = proj_analytical @ torch.diag(eig_val_analytical) @ proj_analytical.t()

    low_rank_approx = lanzcos_low_rank_hessian_approx(lambda z: reg_H_analytical @ z,
                                                      reg_H_analytical.shape,
                                                      rank_estimate=rank_estimate)

    D = torch.diag(low_rank_approx.eigen_vals)
    proj = low_rank_approx.projections

    assert torch.allclose((proj @ D @ proj.t()) @ vec, low_rank_approx_analytical @ vec, rtol=1e-1)

import numpy as np
import torch
from torch.func import functional_call
import dask.array as da
from distributed import Client, LocalCluster
import dask

class LossGrad:
    def __init__(self, model, loss):
        self.loss = loss
        self.model = model

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _compute_single_loss(self, params, x, y):
        outputs = functional_call(self.model, params, (x.unsqueeze(0).to(self.device)))
        return self.loss(outputs, y.unsqueeze(0).to(self.device))

    @staticmethod
    def _flatten_tensors(tensors):
        return torch.cat([t.reshape(t.shape[0], -1) for t in tensors], dim=-1)

    def grads(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        params = {k: p.detach() for k, p in self.model.named_parameters()}
        gradient_fnc = torch.func.jacrev(torch.vmap(self._compute_single_loss, in_dims=(None, 0, 0)))
        return torch.clamp(self._flatten_tensors(gradient_fnc(params, x.to(self.device), y.to(self.device)).values()), -1, 1)

    def grads_dot(self, x_test: torch.Tensor, y_test: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
        _grads_left = self.grads(x_test, y_test)
        _grads_right = self.grads(x, y)
        return torch.einsum("ij,kj->ik", _grads_left, _grads_right)

    def to(self, device):
        self.model = self.model.to(device)
        return self


def block_grads(x, y, _loss_grad: LossGrad):
    t_grads = _loss_grad.grads(torch.from_numpy(x), torch.from_numpy(y))
    return t_grads.cpu().numpy()


def block_grads_dot(x_test, y_test, x, y, _loss_grad: LossGrad):

    t_grads_dot = _loss_grad.grads_dot(torch.from_numpy(x_test), torch.from_numpy(y_test),
                                       torch.from_numpy(x), torch.from_numpy(y))
    return t_grads_dot.cpu().numpy()


if __name__ == "__main__":
    dimensions = (int(1e+5), 10)
    num_params = (dimensions[0] + 1) * dimensions[1]
    num_data = 10000
    num_test_data = 100
    chunk_size = 200
    test_chunk_size = 10
    t_x = torch.rand(num_data, dimensions[0])
    t_y = torch.rand(num_data, dimensions[1])
    #t_x_test = torch.rand(num_test_data, dimensions[0])
    #t_y_test = torch.rand(num_test_data, dimensions[1])


    with Client(LocalCluster(processes=False)) as client:
        torch_model = torch.nn.Linear(*dimensions, bias=True).eval().share_memory()
        loss_grad = LossGrad(torch_model, torch.nn.functional.mse_loss)
        da_x = da.from_array(t_x.numpy(), chunks=(chunk_size, -1))
        #da_x = da.random.random((num_data, dimensions[0]), chunks=(chunk_size, -1)).astype(np.float32)
        #da_y = da.random.random((num_data, dimensions[1]), chunks=(chunk_size, -1)).astype(np.float32)
        da_y = da.from_array(t_y.numpy(), chunks=(chunk_size, -1))
        #da_x_test = da.random.random((num_test_data, dimensions[0]), chunks=(test_chunk_size, -1)).astype(np.float32)

        #da_y_test = da.random.random((num_test_data, dimensions[1]), chunks=(test_chunk_size, -1)).astype(np.float32)
        #da_y_test = da.from_array(t_y_test.numpy(), chunks=(test_chunk_size, -1))
        #da_x_test = da.from_array(t_x_test.numpy(), chunks=(test_chunk_size, -1))
        #da_y_test = da.from_array(t_y_test.numpy(), chunks=(test_chunk_size, -1))

        # ??? what to do, if I want to have the model on gpu?
        loss_grad_future = client.scatter(loss_grad, broadcast=True)
        # only works when chunk_size divides num_data

        #grads_left = da.map_blocks(block_grads, da_x_test, da_y_test, loss_grad_future, dtype=da_x.dtype, chunks=(test_chunk_size, num_params))
        grads_right = da.map_blocks(block_grads, da_x, da_y, loss_grad_future, dtype=da_x.dtype, chunks=(chunk_size, num_params))
        #grads_dot = da.einsum("ij,kj->ik", grads_left, grads_right)
        #grads_dot_block = da.blockwise(block_grads_dot, "ij", da_x_test, "ik", da_y_test, "im", da_x, "jk",
        #                     da_y, "jm", _loss_grad=loss_grad_future, dtype=da_x.dtype, concatenate=True)
        result = da.to_zarr(grads_right, "grads.zarr", overwrite=True)
        #result, result_block = dask.compute(grads_dot, grads_dot_block)
        #numpy_result = np.einsum("ik, jk -> ij", result_grad_left, result_grad_right)
        #assert np.allclose(result, result_block, rtol=1e-3)
        client.close()
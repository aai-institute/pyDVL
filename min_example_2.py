from math import prod

import torch
from torch.func import functional_call
import dask.array as da
from distributed import Client, LocalCluster
from torch.utils.data import Dataset


class LossGrad:
    def __init__(self, model, loss):
        self.loss = loss
        self.model = model

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _compute_single_loss(self, params, x, y):
        outputs = functional_call(self.model, params, (x.unsqueeze(0).to(self.device)))
        return self.loss(outputs, y.unsqueeze(0))

    def apply(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        params = {k: p.detach() for k, p in self.model.named_parameters()}
        gradient_fnc = torch.func.jacrev(torch.vmap(self._compute_single_loss, in_dims=(None, 0, 0)))
        return gradient_fnc(params, x.to(self.device), y.to(self.device))

    def to(self, device):
        self.model = self.model.to(device)
        return self


def block_apply(x, y, _loss_grad: LossGrad):
    tensor_dict = _loss_grad.apply(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    return torch.cat([t.reshape(t.shape[0], -1) for t in tensor_dict.values()], dim=-1).cpu().numpy()


if __name__ == "__main__":
    dimensions = (int(1e5), 1)
    num_params = (dimensions[0] + 1) * dimensions[1]
    num_data = 4*int(1e2)
    chunk_size = 100
    t_x = torch.rand(num_data, dimensions[0])
    t_y = torch.rand(num_data, dimensions[1])

    with Client(LocalCluster(processes=True)) as client:
        torch_model = torch.nn.Linear(*dimensions, bias=True).eval().share_memory()
        loss_grad = LossGrad(torch_model, torch.nn.functional.mse_loss)
        da_x = da.from_array(t_x.numpy(), chunks=(chunk_size, 10000))
        da_y = da.from_array(t_y.numpy(), chunks=(chunk_size, -1))
        # ??? what to do, if I want to have the model on gpu?
        loss_grad_future = client.scatter(loss_grad, broadcast=True)
        # only works when chunk_size divides num_data
        grads = da.map_blocks(block_apply, da_x, da_y, loss_grad_future, dtype=da_x.dtype, chunks=(chunk_size, num_params))
        result = da.to_zarr(grads, "grads.zarr", overwrite=True)
        mem_usage = 4 * prod(grads.shape) / 1000**3


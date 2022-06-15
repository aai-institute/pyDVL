import numpy as np
import torch
import torch.nn as nn


class LRTorchModel(nn.Module):
    def __init__(self, n_input: int, n_output: int, init: np.ndarray = None):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        if init is None:
            init = np.random.normal(0, 0.02, size=(n_output, n_input))

        tt_A = torch.tensor(init, dtype=torch.float32)
        self.A = nn.Parameter(tt_A, requires_grad=True)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x @ self.A.T

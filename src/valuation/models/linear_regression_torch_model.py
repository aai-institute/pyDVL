import numpy as np
import torch
import torch.nn as nn


class LRTorchModel(nn.Module):
    def __init__(self, n_input: int, n_output: int):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        tt_A = torch.tensor(
            np.random.normal(0, 0.02, size=(n_output, n_input)), dtype=torch.float32
        )
        self.A = nn.Parameter(tt_A, requires_grad=True)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x @ self.A.T

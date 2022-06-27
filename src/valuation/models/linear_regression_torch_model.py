from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    pass


class LRTorchModel(nn.Module):
    def __init__(self, dim=Tuple[int, int], init: Tuple[np.ndarray, np.ndarray] = None):
        super().__init__()
        n_output, n_input = dim
        self.n_input = n_input
        self.n_output = n_output
        if init is None:
            init_A = np.random.normal(0, 0.02, size=(n_output, n_input))
            init_b = np.random.normal(0, 0.02, size=(n_output))
            init = (init_A, init_b)

        self.A = nn.Parameter(
            torch.tensor(init[0], dtype=torch.float32), requires_grad=True
        )
        self.b = nn.Parameter(
            torch.tensor(init[1], dtype=torch.float32), requires_grad=True
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x @ self.A.T + self.b

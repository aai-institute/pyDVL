# FIXME the following code was part of an attempt to accommodate different
# frameworks. In its current form it is ugly and thus it will likely be changed
# in the future.

from .twice_differentiable import TwiceDifferentiable

__all__ = ["TwiceDifferentiable"]

try:
    import torch

    from .torch_differentiable import TorchTwiceDifferentiable

    __all__.append("TorchTwiceDifferentiable")

    from .torch_differentiable import (
        as_tensor,
        einsum,
        mvp,
        solve_batch_cg,
        solve_linear,
        solve_lissa,
        stack,
    )

    TensorType = torch.Tensor
    ModelType = torch.nn.Module

except ImportError:
    pass

__all__.extend(
    [
        "TensorType",
        "ModelType",
        "solve_linear",
        "solve_batch_cg",
        "solve_lissa",
        "as_tensor",
        "stack",
        "einsum",
        "mvp",
    ]
)

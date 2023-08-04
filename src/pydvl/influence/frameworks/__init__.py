# FIXME the following code was part of an attempt to accommodate different
# frameworks. In its current form it is ugly and thus it will likely be changed
# in the future.

import logging

from .twice_differentiable import IhvpResult, TwiceDifferentiable

__all__ = ["TwiceDifferentiable"]
logger = logging.getLogger("frameworks")

try:
    import torch

    from .torch_differentiable import TorchTwiceDifferentiable

    __all__.append("TorchTwiceDifferentiable")

    from .torch_differentiable import (
        as_tensor,
        cat,
        einsum,
        mvp,
        solve_batch_cg,
        solve_linear,
        solve_lissa,
        stack,
        transpose_tensor,
        zero_tensor,
    )

    TensorType = torch.Tensor
    DataLoaderType = torch.utils.data.DataLoader
    ModelType = torch.nn.Module

    __all__.extend(
        [
            "TensorType",
            "ModelType",
            "IhvpResult",
            "solve_linear",
            "solve_batch_cg",
            "solve_lissa",
            "as_tensor",
            "stack",
            "cat",
            "zero_tensor",
            "transpose_tensor",
            "einsum",
            "mvp",
        ]
    )

except ImportError:
    logger.info(
        "No compatible framework found. Influence function computation disabled."
    )

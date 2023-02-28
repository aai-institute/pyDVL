import logging
import os

from .base_twice_differentiable import BaseTwiceDifferentiable

logger = logging.getLogger(__name__)

available_frameworks = []
try:
    import torch

    available_frameworks.append("torch")
except ImportError:
    pass


if available_frameworks == []:
    raise ValueError("No supported framework is installed. Currently supported: torch.")
elif len(available_frameworks) > 1:
    logger.warning(
        f"There are multiple available frameworks: {available_frameworks}. "
        f"Environment choice will default to {available_frameworks[0]}"
        f"You can select the one you want to use by setting os.environ['FRAMEWORK']= preferred_framework"
    )

FRAMEWORK = os.environ.get("FRAMEWORK", "Not Set")
if FRAMEWORK == "Not Set":
    FRAMEWORK = available_frameworks[0]

if FRAMEWORK == "torch":
    from .torch_differentiable import TorchTwiceDifferentiable as TwiceDifferentiable

    TensorType = torch.Tensor
    # etc. (or something like that)
else:
    raise ValueError(f"Unknown framework: {FRAMEWORK}")

__all__ = [
    "TwiceDifferentiable",
]

from opt_einsum import contract

from valuation.models.pytorch_model import TwiceDifferentiable
from valuation.utils import Utility
from valuation.utils.algorithms import conjugate_gradient
from valuation.utils.types import BatchInfluenceFunction


def influence_functions(utility: Utility, progress: bool = False) -> BatchInfluenceFunction:
    model = utility.model
    data = utility.data

    if not issubclass(model.__class__, TwiceDifferentiable):
        raise AttributeError("Model is not twice differentiable, please implement interface.")

    twd: TwiceDifferentiable = model
    grad = twd.grad(data.x_train, data.y_train, progress=progress)
    hvp = lambda v: twd.hvp(data.x_train, data.y_train, v, progress=progress)
    influence_factors = conjugate_gradient(hvp, grad)[0]
    return lambda v: contract('ia,ja->ij', v, influence_factors)

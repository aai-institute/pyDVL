from opt_einsum import contract

from valuation.models.pytorch_model import TwiceDifferentiable
from valuation.utils import Utility
from valuation.utils.algorithms import conjugate_gradient
from valuation.utils.types import BatchInfluenceFunction


def calculate_batched_influence_functions(utility: Utility, progress: bool = False) -> BatchInfluenceFunction:
    """
    Calculates the influence functions I(v), which can be used to obtain the influences for data samples v. It does
    so by calculating the gradient of the loss for each data sample with respect to the parameters. Subsequently it
    gets multiplied with the inverse hessian from the left, using conjugate gradient along with hessian vector products
    from the TwiceDifferentiable interface.

    :param utility: Utility object with model, data, and scoring function. The model has to inherit from the
    TwiceDifferentiable interface.
    :param progress: whether to display progress bars
    :returns: A function which takes a vector of size [M, K] and outputs a matrix of [M, N]. M is the number of test
    samples, N is the number of train samples and K is the number of features in the input space.
    """
    model = utility.model
    data = utility.data

    if not issubclass(model.__class__, TwiceDifferentiable):
        raise AttributeError("Model is not twice differentiable, please implement interface.")

    twd: TwiceDifferentiable = model
    grad = twd.grad(data.x_train, data.y_train, progress=progress)
    hvp = lambda v: twd.hvp(data.x_train, data.y_train, v, progress=progress)
    influence_factors = conjugate_gradient(hvp, grad)[0]
    return lambda v: contract('ia,ja->ij', v, influence_factors)

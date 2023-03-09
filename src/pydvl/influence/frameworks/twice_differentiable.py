from abc import ABC
from typing import Callable, Generic, List, Sequence, Tuple, TypeVar

TensorType = TypeVar("TensorType", bound=Sequence)
ModelType = TypeVar("ModelType")


class TwiceDifferentiable(ABC, Generic[TensorType, ModelType]):
    """Wraps a differentiable model and loss and provides methods to compute the
    second derivative of the loss wrt. the model parameters.
    """

    def __init__(
        self,
        model: ModelType,
        loss: Callable[[TensorType, TensorType], TensorType],
    ):
        pass

    def num_params(self) -> int:
        """Returns the number of parameters of the model"""
        pass

    def parameters(self) -> List[TensorType]:
        """Returns all the model parameters that require differentiation"""
        pass

    def split_grad(
        self,
        x: TensorType,
        y: TensorType,
        progress: bool = False,
    ) -> TensorType:
        """
        Calculate the gradient of the model wrt each input x and labels y.
        The output is therefore of size [Nxp], with N the amount of points (the
        length of x and y) and P the number of parameters.

        :param x: An array representing the features $x_i$.
        :param y: An array representing the predicted target values $y_i$.
        :param progress: True, iff progress shall be printed.
        :returns: An array representing the gradients wrt. the parameters of the
            model.
        """
        pass

    def grad(self, x: TensorType, y: TensorType) -> Tuple[TensorType, TensorType]:
        """
        Calculates gradient of model parameters wrt. the model parameters.
        :param x: A matrix representing the features $x_i$.
        :param y: A matrix representing the target values $y_i$.
        :returns: A tuple where: \
            - first element is an array with the gradients of the model. \
            - second element is the input to the model as a grad parameters. \
                This can be used for further differentiation. 
        """
        pass

    def hessian(
        self,
        x: TensorType,
        y: TensorType,
        progress: bool = False,
    ) -> TensorType:
        """Calculates the full Hessian of $L(f(x),y)$ with respect to the model
        parameters given data ($x$ and $y$).
        :param x: An array representing the features $x_i$.
        :param y: An array representing the target values $y_i$.
        :returns: the hessian of the model, i.e. the second derivative wrt. the
            model parameters.
        """
        pass

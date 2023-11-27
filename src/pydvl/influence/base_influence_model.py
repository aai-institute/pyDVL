from abc import ABC, abstractmethod
from enum import Enum
from typing import Collection, Generic, Optional, TypeVar


class InfluenceType(str, Enum):
    r"""
    Enum representation for the types of influence.

    Attributes:
        Up: Up-weighting a training point, see section 2.1 of
            (Koh and Liang, 2017)<sup><a href="#koh_liang_2017">1</a></sup>
        Perturbation: Perturb a training point, see section 2.2 of
            (Koh and Liang, 2017)<sup><a href="#koh_liang_2017">1</a></sup>

    """

    Up = "up"
    Perturbation = "perturbation"


class UnSupportedInfluenceTypeException(ValueError):
    def __init__(self, influence_type: str):
        super().__init__(
            f"Provided {influence_type=} is not supported. Choose one of InfluenceType.Up "
            f"and InfluenceType.Perturbation"
        )


"""Type variable for tensors, i.e. sequences of numbers"""
TensorType = TypeVar("TensorType", bound=Collection)


class Influence(Generic[TensorType], ABC):
    """
    Generic abstract base class for computing influence related quantities. For a specific influence algorithm and
    tensor framework, inherit from this base class
    """

    @property
    @abstractmethod
    def num_parameters(self):
        """Number of trainable parameters of the underlying model"""

    @abstractmethod
    def up_weighting(
        self,
        z_test_factors: TensorType,
        x: TensorType,
        y: TensorType,
    ) -> TensorType:
        r"""
        Overwrite this method to implement the computation of
        $$
        \langle z_test_factors, \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle
        $$
        where the gradient is meant to be per sample of the batch $(x, y)$.
        Args:
            z_test_factors: pre-computed array, approximating $H^{-1}\nabla_{\theta} \ell(y_{test}, f_{\theta}(x_{test}))$
            x: model input to use in the gradient computations
            y: label tensor to compute gradients

        Returns:
            Tensor representing the element-wise scalar product of the provided batch

        """

    @abstractmethod
    def perturbation(
        self,
        z_test_factors: TensorType,
        x: TensorType,
        y: TensorType,
    ) -> TensorType:
        r"""
        Overwrite this method to implement the computation of
        $$
        \langle z_test_factors, \nabla_x \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle
        $$
        where the gradient is meant to be per sample of the batch $(x, y)$.
        Args:
            z_test_factors: pre-computed array, approximating $H^{-1}\nabla_{\theta} \ell(y_{test}, f_{\theta}(x_{test}))$
            x: model input to use in the gradient computations
            y: label tensor to compute gradients

        Returns:
            Tensor representing the element-wise scalar product for the provided batch

        """

    @abstractmethod
    def factors(self, x: TensorType, y: TensorType) -> TensorType:
        r"""
        Overwrite this method to implement the approximation of
        $$
        H^{-1}\nabla_{theta} \ell(y, f_{\theta}(x))
        $$
        where the gradient is meant to be per sample of the batch $(x, y)$.

        Args:
            x: model input to use in the gradient computations
            y: label tensor to compute gradients

        Returns:
            Tensor representing the element-wise inverse Hessian matrix vector products

        """

    @abstractmethod
    def values(
        self,
        x_test: TensorType,
        y_test: TensorType,
        x: Optional[TensorType] = None,
        y: Optional[TensorType] = None,
        influence_type: InfluenceType = InfluenceType.Up,
    ) -> TensorType:
        r"""
        Overwrite this method to implement the approximation of
        $$
        \langle H^{-1}\nabla_{theta} \ell(y_{test}, f_{\theta}(x_{test})), \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle
        $$
        for the case of up-weighting influence, resp.
        $$
        \langle H^{-1}\nabla_{theta} \ell(y_{test}, f_{\theta}(x_{test})), \nabla_{x} \nabla_{\theta} \ell(y, f_{\theta}(x)) \rangle
        $$
        for the perturbation type influence case.

        Args:
            x_test: model input to use in the gradient computations of $H^{-1}\nabla_{theta} \ell(y_{test}, f_{\theta}(x_{test}))$
            y_test: label tensor to compute gradients
            x: optional model input to use in the gradient computations $\nabla_{theta}\ell(y, f_{\theta}(x))$, resp. $\nabla_{x}\nabla_{theta}\ell(y, f_{\theta}(x))$, if None,
                use $x=x_{test}$
            y: optional label tensor to compute gradients
            influence_type: enum value of [InfluenceType][pydvl.influence.twice_differentiable.InfluenceType]

        Returns:
            Tensor representing the element-wise scalar products for the provided batch

        """

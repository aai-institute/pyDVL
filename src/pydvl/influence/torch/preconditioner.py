from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

from ...utils.exceptions import NotFittedException
from .functional import LowRankProductRepresentation, operator_nystroem_approximation

if TYPE_CHECKING:
    from .operator import TensorOperator

__all__ = ["JacobiPreconditioner", "NystroemPreconditioner", "Preconditioner"]


class Preconditioner(ABC):
    r"""
    Abstract base class for implementing pre-conditioners for improving the convergence
    of CG for systems of the form

    \[ ( A + \lambda \operatorname{I})x = \operatorname{rhs} \]

    i.e. a matrix $M$ such that $M^{-1}(A + \lambda \operatorname{I})$ has a better
    condition number than $A + \lambda \operatorname{I}$.

    """

    _reg: Optional[float]

    @property
    def regularization(self):
        return self._reg

    @regularization.setter
    def regularization(self, value: float):
        if self.modify_regularization_requires_fit:
            raise NotImplementedError(
                f"Adapting regularization for instances of type "
                f"{type(self)} without re-fitting is not "
                f"supported. Call the fit method instead."
            )
        self._validate_regularization(value)
        self._reg = value

    def _validate_regularization(self, value: Optional[float]):
        if value is not None and value < 0:
            raise ValueError("regularization must be non-negative")

    @property
    @abstractmethod
    def modify_regularization_requires_fit(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_fitted(self):
        pass

    def fit(
        self,
        operator: "TensorOperator",
        regularization: Optional[float] = None,
    ):
        r"""
        Implement this to fit the pre-conditioner to the matrix represented by the
        mat_mat_prod
        Args:
            operator: The preconditioner is fitted to this operator
            regularization: regularization parameter $\lambda$ in the equation
                $ ( A + \lambda \operatorname{I})x = \operatorname{rhs} $
        Returns:
            self
        """
        self._validate_regularization(regularization)
        return self._fit(operator, regularization)

    @abstractmethod
    def _fit(self, operator: "TensorOperator", regularization: Optional[float] = None):
        pass

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        r"""
        Solve the equation $M@Z = \operatorname{rhs}$
        Args:
            rhs: right hand side of the equation, corresponds to the residuum vector
                (or matrix) in the conjugate gradient method

        Returns:
            solution $M^{-1}\operatorname{rhs}$

        """
        if not self.is_fitted:
            raise NotFittedException(type(self))

        return self._solve(rhs)

    @abstractmethod
    def _solve(self, rhs: torch.Tensor):
        pass

    @abstractmethod
    def to(self, device: torch.device) -> Preconditioner:
        """Implement this to move the (potentially fitted) preconditioner to a
        specific device"""


class JacobiPreconditioner(Preconditioner):
    r"""
    Pre-conditioner for improving the convergence of CG for systems of the form

    $$ ( A + \lambda \operatorname{I})x = \operatorname{rhs} $$

    The JacobiPreConditioner uses the diagonal information of the matrix $A$.
    The diagonal elements are not computed directly but estimated via Hutchinson's
    estimator.

    $$ M = \frac{1}{m} \sum_{i=1}^m u_i \odot Au_i + \lambda \operatorname{I} $$

    where $u_i$ are i.i.d. Gaussian random vectors.
    Works well in the case the matrix $A + \lambda \operatorname{I}$ is diagonal
    dominant.
    For more information, see the documentation of
    [Conjugate Gradient][conjugate-gradient]
    Args:
        num_samples_estimator: number of samples to use in computation of
            Hutchinson's estimator
    """

    _diag: torch.Tensor

    def __init__(self, num_samples_estimator: int = 1):
        self.num_samples_estimator = num_samples_estimator

    @property
    def is_fitted(self):
        has_diag = hasattr(self, "_diag") and self._diag is not None
        has_regularization = hasattr(self, "_reg")
        return has_diag and has_regularization

    @property
    def modify_regularization_requires_fit(self) -> bool:
        return False

    def _fit(
        self,
        operator: "TensorOperator",
        regularization: Optional[float] = None,
    ):
        r"""
        Fits by computing an estimate of the diagonal of the matrix represented by
        a [TensorOperator][pydvl.influence.torch.base.TensorOperator]
        via Hutchinson's estimator

        Args:
            operator: The preconditioner is fitted to this operator
            regularization: regularization parameter
                $\lambda$ in $(A+\lambda I)x=b$
        """
        random_samples = torch.randn(
            self.num_samples_estimator,
            operator.input_size,
            device=operator.device,
            dtype=operator.dtype,
        )

        diagonal_estimate = torch.sum(
            torch.mul(random_samples, operator.apply(random_samples)), dim=0
        )

        diagonal_estimate /= self.num_samples_estimator
        self._diag = diagonal_estimate
        self._reg = regularization

    def _solve(self, rhs: torch.Tensor):
        diag = self._diag

        if self._reg is not None:
            diag = diag + self._reg

        inv_diag = 1.0 / diag

        if rhs.ndim == 1:
            return rhs * inv_diag

        return rhs * inv_diag.unsqueeze(-1)

    def to(self, device: torch.device) -> JacobiPreconditioner:
        if self._diag is not None:
            self._diag = self._diag.to(device)
        return self


class NystroemPreconditioner(Preconditioner):
    r"""
    Pre-conditioner for improving the convergence of CG for systems of the form

    $$ (A + \lambda \operatorname{I})x = \operatorname{rhs} $$

    The NystroemPreConditioner computes a low-rank approximation

    $$ A_{\text{nys}} = (A \Omega)(\Omega^T A \Omega)^{\dagger}(A \Omega)^T
    = U \Sigma U^T, $$

    where $(\cdot)^{\dagger}$ denotes the [Moore-Penrose inverse](
    https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse),
    and uses the matrix

    $$ M^{-1} = (\lambda + \sigma_{\text{rank}})U(\Sigma+
        \lambda \operatorname{I})^{-1}U^T+(\operatorname{I} - UU^T) $$

    for pre-conditioning, where \(  \sigma_{\text{rank}} \) is the smallest
    eigenvalue of the low-rank approximation.
    """

    _low_rank_approx: LowRankProductRepresentation

    def __init__(self, rank: int):
        self._rank = rank

    @property
    def low_rank_approx(self) -> Optional[LowRankProductRepresentation]:
        return self._low_rank_approx

    @property
    def rank(self):
        return self._rank

    @property
    def is_fitted(self):
        has_low_rank_approx = (
            hasattr(self, "_low_rank_approx") and self._low_rank_approx is not None
        )
        has_regularization = hasattr(self, "_reg") and self._reg is not None
        return has_low_rank_approx and has_regularization

    @property
    def modify_regularization_requires_fit(self) -> bool:
        return False

    def _fit(
        self,
        operator: "TensorOperator",
        regularization: Optional[float] = None,
    ):
        r"""
        Fits by computing a low-rank approximation of the matrix represented by
        `mat_mat_prod` via Nystroem approximation

        Args:
            operator: The preconditioner is fitted to this operator
            regularization: regularization parameter
                $\lambda$  in $(A+\lambda I)x=b$
        """

        self._low_rank_approx = operator_nystroem_approximation(operator, self._rank)
        self._reg = regularization

    def _solve(self, rhs: torch.Tensor):
        rhs_is_one_dim = rhs.ndim == 1
        b = torch.atleast_2d(rhs).t() if rhs_is_one_dim else rhs

        U = self._low_rank_approx.projections

        Sigma = self._low_rank_approx.eigen_vals
        lambda_rank = self._low_rank_approx.eigen_vals[-1]

        if self._reg is not None:
            Sigma = Sigma + self._reg
            lambda_rank = lambda_rank + self._reg

        U_t_b = U.t() @ b

        Sigma_inv = lambda_rank / Sigma

        result = U @ (U_t_b * Sigma_inv.unsqueeze(-1))
        result += b - U @ U_t_b

        if rhs_is_one_dim:
            result = result.squeeze()

        return result

    def to(self, device: torch.device) -> NystroemPreconditioner:
        if self._low_rank_approx is not None:
            self._low_rank_approx = self._low_rank_approx.to(device)
        return self

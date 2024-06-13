from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

from ..base_influence_function_model import NotFittedException
from .functional import LowRankProductRepresentation, operator_nystroem_approximation

if TYPE_CHECKING:
    from .operator import TensorOperator

__all__ = ["JacobiPreConditioner", "NystroemPreConditioner", "PreConditioner"]


class PreConditioner(ABC):
    r"""
    Abstract base class for implementing pre-conditioners for improving the convergence
    of CG for systems of the form

    \[ ( A + \lambda \operatorname{I})x = \operatorname{rhs} \]

    i.e. a matrix $M$ such that $M^{-1}(A + \lambda \operatorname{I})$ has a better
    condition number than $A + \lambda \operatorname{I}$.

    """

    @property
    @abstractmethod
    def is_fitted(self):
        pass

    @abstractmethod
    def fit(
        self,
        operator: "TensorOperator",
        regularization: float = 0.0,
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
        pass

    def solve(self, rhs: torch.Tensor):
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
    def to(self, device: torch.device) -> PreConditioner:
        """Implement this to move the (potentially fitted) preconditioner to a
        specific device"""


class JacobiPreConditioner(PreConditioner):
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
    _reg: float

    def __init__(self, num_samples_estimator: int = 1):
        self.num_samples_estimator = num_samples_estimator

    @property
    def is_fitted(self):
        return self._diag is not None and self._reg is not None

    def fit(
        self,
        operator: "TensorOperator",
        regularization: float = 0.0,
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
            operator.input_size,
            self.num_samples_estimator,
            device=operator.device,
            dtype=operator.dtype,
        )
        diagonal_estimate = torch.sum(
            torch.mul(random_samples, operator.apply(random_samples)), dim=1
        )
        diagonal_estimate /= self.num_samples_estimator
        self._diag = diagonal_estimate
        self._reg = regularization

    def _solve(self, rhs: torch.Tensor):
        inv_diag = 1.0 / (self._diag + self._reg)

        if rhs.ndim == 1:
            return rhs * inv_diag

        return rhs * inv_diag.unsqueeze(-1)

    def to(self, device: torch.device) -> JacobiPreConditioner:
        if self._diag is not None:
            self._diag = self._diag.to(device)
        return self


class NystroemPreConditioner(PreConditioner):
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
    _regularization: float

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
        return self._low_rank_approx is not None and self._regularization is not None

    def fit(
        self,
        operator: "TensorOperator",
        regularization: float = 0.0,
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
        self._regularization = regularization

    def _solve(self, rhs: torch.Tensor):

        rhs_is_one_dim = rhs.ndim == 1
        b = torch.atleast_2d(rhs).t() if rhs_is_one_dim else rhs

        U = self._low_rank_approx.projections

        Sigma = self._low_rank_approx.eigen_vals + self._regularization
        lambda_rank = self._low_rank_approx.eigen_vals[-1] + self._regularization

        U_t_b = U.t() @ b

        Sigma_inv = lambda_rank / Sigma

        result = U @ (U_t_b * Sigma_inv.unsqueeze(-1))
        result += b - U @ U_t_b

        if rhs_is_one_dim:
            result = result.squeeze()

        return result

    def to(self, device: torch.device) -> NystroemPreConditioner:
        if self._low_rank_approx is not None:
            self._low_rank_approx = self._low_rank_approx.to(device)
        return self

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch

from ..base_influence_function_model import NotFittedException
from .functional import LowRankProductRepresentation, randomized_nystroem_approximation


class PreConditioner(ABC):
    @property
    @abstractmethod
    def is_fitted(self):
        pass

    @abstractmethod
    def fit(
        self,
        mat_mat_prod: Callable[[torch.Tensor], torch.Tensor],
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        regularization: float = 0.0,
    ):
        pass

    def solve(self, rhs: torch.Tensor):

        if not self.is_fitted:
            raise NotFittedException(type(self))

        return self._solve(rhs)

    @abstractmethod
    def _solve(self, rhs: torch.Tensor):
        pass


class JacobiPreConditioner(PreConditioner):
    r"""
    Pre-conditioner for improving the convergence of CG for systems of the form

    \[ (A + \lambda \operatorname{I})x = \operatorname{rhs} \]

    The JacobiPreConditioner uses the diagonal information of the matrix $A$.
    The diagonal elements are not computed directly but estimated via Hutchinson's
    estimator,

    \[ \frac{1}{m} \sum_{i=1}^m u_i \odot Au_i\]

    Works well in the case the matrix \(A\) is diagonal dominant.
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
        mat_mat_prod: Callable[[torch.Tensor], torch.Tensor],
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        regularization: float = 0.0,
    ):
        r"""
        Computing an estimate of the diagonal of the matrix represented by
        the mat_mat_prod via Hutchinson's estimator

        Args:
            mat_mat_prod: a callable representing the matrix-matrix product
            size: size of the square matrix
            dtype: needed data type of inputs for the mat_mat_prod
            device: needed device for inputs of mat_mat_prod
            regularization: regularization parameter
                \(\lambda\) in \((A+\lambda I)x=b\)
        """
        random_samples = torch.randn(
            size, self.num_samples_estimator, device=device, dtype=dtype
        )
        diagonal_estimate = torch.sum(
            torch.mul(random_samples, mat_mat_prod(random_samples)), dim=1
        )
        diagonal_estimate /= self.num_samples_estimator
        self._diag = diagonal_estimate
        self._reg = regularization

    def _solve(self, rhs: torch.Tensor):
        inv_diag = 1.0 / (self._diag + self._reg)

        if rhs.ndim == 1:
            return rhs * inv_diag

        return rhs * inv_diag.unsqueeze(-1)


class NystroemPreConditioner(PreConditioner):
    r"""
    Pre-conditioner for improving the convergence of CG for systems of the form

    \[ (A + \lambda \operatorname{I})x = \operatorname{rhs} \]

    The NystroemPreConditioner computes a low-rank approximation

    \[ A_{\text{nys}} = (A \Omega)(\Omega^T A \Omega)^{+}(A \Omega)^T
    = U \Sigma U^T \]

    and uses the matrix

    \[ P^{-1} = (\lambda + \sigma_{\text{rank}})U(\Sigma+
        \lambda \operatorname{I})^{-1}U^T+(\operatorname{I} - UU^T) \]

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
        mat_mat_prod: Callable[[torch.Tensor], torch.Tensor],
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        regularization: float = 0.0,
    ):
        r"""
        Computing a low-rank approximation of the matrix represented by `mat_mat_prod`
        via Nystroem approximation

        Args:
            mat_mat_prod: a callable representing the matrix-matrix product
            size: size of the square matrix
            dtype: needed data type of inputs for the mat_mat_prod
            device: needed device for inputs of mat_mat_prod
            regularization: regularization parameter
                \(\lambda\) in \((A+\lambda I)x=b\)
        """

        self._low_rank_approx = randomized_nystroem_approximation(
            mat_mat_prod, size, self._rank, dtype, mat_vec_device=device
        )
        self._regularization = regularization

    def _solve(self, rhs: torch.Tensor):
        regularized_eigenvalues = (
            self._low_rank_approx.eigen_vals + self._regularization
        )
        lambda_rank = self._low_rank_approx.eigen_vals[-1] + self._regularization

        proj_rhs = self._low_rank_approx.projections.t() @ (
            rhs.unsqueeze(-1) if rhs.ndim == 1 else rhs
        )

        inverse_regularized_eigenvalues = lambda_rank / regularized_eigenvalues

        result = self._low_rank_approx.projections @ (
            proj_rhs * inverse_regularized_eigenvalues.unsqueeze(-1)
        )

        result += (
            rhs.unsqueeze(-1) if rhs.ndim == 1 else rhs
        ) - self._low_rank_approx.projections @ proj_rhs
        return result.squeeze()

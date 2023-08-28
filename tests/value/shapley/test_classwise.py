"""
Test cases for the class wise shapley value.
"""
import random
from random import seed
from typing import Dict, Tuple, cast

import numpy as np
import pandas as pd
import pytest
from numpy._typing import NDArray

from pydvl.utils import Dataset, Utility, powerset
from pydvl.value import MaxChecks, ValuationResult
from pydvl.value.shapley.classwise import (
    ClasswiseScorer,
    compute_classwise_shapley_values,
)
from pydvl.value.shapley.truncated import NoTruncation
from tests.value import check_values


@pytest.fixture(scope="function")
def linear_classifier_cs_scorer_args_exact_solution_use_default_score() -> Tuple[
    Dict, ValuationResult, Dict
]:
    r"""
    Returns the exact solution for the class wise shapley value of the training and
    validation set of the `utility_alt_seq_cf_linear_classifier_cs_scorer` fixture.

    ===========================
    CS-Shapley Manual Derivation
    ===========================

    :Author: Markus Semmler
    :Date:   August 2023

    Dataset description
    ===================

    We have a training and a test dataset. We want to model a simple XOR dataset. The
    development set :math:`D` is given by

    .. math::
        \begin{aligned}
            \hat{x}_0 &= 1 \quad &\hat{y}_0 = 0 \\
            \hat{x}_1 &= 2 \quad &\hat{y}_1 = 0 \\
            \hat{x}_2 &= 3 \quad &\hat{y}_2 = 0 \\
            \hat{x}_3 &= 4 \quad &\hat{y}_3 = 1 \\
        \end{aligned}

    and the training set :math:`T` is given by

    .. math::
        \begin{aligned}
            x_0 &= 1 \quad &y_0 = 0 \\
            x_1 &= 2 \quad &y_1 = 0 \\
            x_2 &= 3 \quad &y_2 = 1 \\
            x_3 &= 4 \quad &y_3 = 1 \\
        \end{aligned}

    Note that the training set and the development set contain the same
    inputs x, but differ in the label :math:`\hat{y}_2 \neq y_2`

    Model
    =====

    We use an adapted version of linear regression

    .. math:: y = \max(0, \min(1, \text{round}(\beta^T x)))

    for classification, with the closed form solution

    .. math:: \beta = \frac{\text{dot}(x, y)}{\text{dot}(x, x)}

    Fitted model
    ============

    The hyperparameters for all combinations are

    .. container:: tabular

       | \|c||Sc \| Sc \| Sc \| Sc \| :math:`S_1 \cup S_2` &
         :math:`\emptyset` & :math:`\{x_2\}` & :math:`\{x_3\}` &
         :math:`\{x_2, x_3\}`
       | :math:`\emptyset` & nan & :math:`\frac{1}{3}` & :math:`\frac{1}{4}`
         & :math:`\frac{7}{25}`
       | :math:`\{x_0\}` & :math:`0` & :math:`\frac{3}{10}` &
         :math:`\frac{4}{17}` & :math:`\frac{7}{26}`
       | :math:`\{x_1\}` & :math:`0` & :math:`\frac{3}{13}` &
         :math:`\frac{1}{5}` &\ :math:`\frac{7}{29}`
       | :math:`\{x_0, x_1 \}` & :math:`0` & :math:`\frac{3}{14}` &
         :math:`\frac{4}{21}` & :math:`\frac{7}{30}`

    Accuracy tables on development set :math:`D`
    ============================================

    (*) Note that the algorithm described in the paper overwrites these
    values with 0.

    .. container:: tabular

       | \|c||Sc \| Sc \| Sc \| Sc \| :math:`S_1 \cup S_2` &
         :math:`\emptyset` & :math:`\{x_2\}` & :math:`\{x_3\}` &
         :math:`\{x_2, x_3\}`
       | :math:`\emptyset` & :math:`0` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{4}` & :math:`\frac{1}{4}`
       | :math:`\{x_0\}` & :math:`\frac{3}{4}` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{2}` & :math:`\frac{1}{4}`
       | :math:`\{x_1\}` & :math:`\frac{3}{4}` & :math:`\frac{1}{2}` &
         :math:`\frac{1}{2}` &\ :math:`\frac{1}{2}`
       | :math:`\{x_0, x_1 \}` & :math:`\frac{3}{4}` & :math:`\frac{1}{2}` &
         :math:`\frac{1}{2}` & :math:`\frac{1}{2}`

    .. container:: tabular

       | \|c||Sc \| Sc \| Sc \| Sc \| :math:`S_1 \cup S_2` &
         :math:`\emptyset` & :math:`\{x_2\}` & :math:`\{x_3\}` &
         :math:`\{x_2, x_3\}`
       | :math:`\emptyset` & :math:`0` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{4}` & :math:`\frac{1}{4}`
       | :math:`\{x_0\}` & :math:`0` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{4}` & :math:`\frac{1}{4}`
       | :math:`\{x_1\}` & :math:`0` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{4}` &\ :math:`\frac{1}{4}`
       | :math:`\{x_0, x_1 \}` & :math:`0` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{4}` & :math:`\frac{1}{4}`

    CS-Shapley
    ==========

    The formulas of the algorithm are given by

    .. math::

        \begin{aligned}
            \delta(\pi, S_{-y_i}, i) &= v_{y_i}(\pi_{:i} \cup \{ i \} | S_{-y_i})
                - v_{y_i}(\pi_{:i} | S_{-y_i}) \\
            \left [ \phi_i | S_{-y_i} \right ] &= \frac{1}{|T_{y_i}|!}
                \sum_{\pi \in \Pi(T_{y_i})} \delta(\pi, S_{-y_i}, i) \\
            \phi_i &= \frac{1}{2^{|T_{-y_i}|}-1} \left [\sum_{\emptyset \subset S_{-y_i}
                \subseteq T_{-y_i}} \left [ \phi_i | S_{-y_i} \right ] \right ]
        \end{aligned}

    Valuation of :math:`x_0`
    ========================

    .. math::
        \begin{aligned}
            \delta((x_0, x_1), \{ x_2 \}, 0) &= \frac{1}{4} e^\frac{1}{4} &\quad
                \delta((x_1, x_0), \{ x_2 \}, 0) &= 0 \\
            \delta((x_0, x_1), \{ x_3 \}, 0) &= \frac{1}{2} e^\frac{1}{4} &\quad
                \delta((x_1, x_0), \{ x_3 \}, 0) &= 0 \\
            \delta((x_0, x_1), \{ x_2, x_3 \}, 0) &= \frac{1}{4} e^\frac{1}{4} &\quad
                \delta((x_1, x_0), \{ x_2, x_3 \}, 0) &= 0
        \end{aligned}

    .. math::
        \begin{aligned}
            \left [ \phi_0 | \{ x_2 \} \right] &= \frac{1}{8} e^\frac{1}{4} \\
            \left [ \phi_0 | \{ x_3 \} \right] &= \frac{1}{4} e^\frac{1}{4} \\
            \left [ \phi_0 | \{ x_2, x_3 \} \right] &= \frac{1}{8} e^\frac{1}{4}
        \end{aligned}

    .. math:: \phi_0 = \frac{1}{6} e^\frac{1}{4} \approx 0.214

    Valuation of :math:`x_1`
    ========================

    .. math::
        \begin{aligned}
            \delta((x_0, x_1), \{ x_2 \}, 1) &= \frac{1}{4} e^\frac{1}{4} &\quad
                \delta((x_1, x_0), \{ x_2 \}, 1) &= \frac{1}{2} e^\frac{1}{4} \\
            \delta((x_0, x_1), \{ x_3 \}, 1) &= 0 &\quad
                \delta((x_1, x_0), \{ x_3 \}, 1) &= \frac{1}{2} e^\frac{1}{4} \\
            \delta((x_0, x_1), \{ x_2, x_3 \}, 1) &= \frac{1}{4} e^\frac{1}{4} &\quad
                \delta((x_1, x_0), \{ x_2, x_3 \}, 1) &= \frac{1}{2} e^\frac{1}{4}
        \end{aligned}

    .. math::
        \begin{aligned}
            \left [ \phi_1 | \{ x_2 \} \right] &= \frac{3}{8} e^\frac{1}{4} \\
            \left [ \phi_1 | \{ x_3 \} \right] &= \frac{1}{4} e^\frac{1}{4} \\
            \left [ \phi_1 | \{ x_2, x_3 \} \right] &= \frac{3}{8} e^\frac{1}{4}
        \end{aligned}

    .. math:: \phi_0 = \frac{1}{3} e^\frac{1}{4} \approx 0.428

    Valuation of :math:`x_2`
    ========================

    .. math::
        \begin{aligned}
            \delta((x_2, x_3), \{ x_0 \}, 2) &= \frac{1}{4} e^\frac{1}{4} &\quad
                \delta((x_3, x_2), \{ x_0 \}, 2)
                &= \frac{1}{4} e^\frac{1}{4} - \frac{1}{4} e^\frac{1}{2} \\
            \delta((x_2, x_3), \{ x_1 \}, 2) &= \frac{1}{4} e^\frac{1}{2} &\quad
                \delta((x_3, x_2), \{ x_1 \}, 2) &= 0 \\
            \delta((x_2, x_3), \{ x_0, x_1 \}, 2) &= \frac{1}{4} e^\frac{1}{2} &\quad
                \delta((x_3, x_2), \{ x_0, x_1 \}, 2) &= 0
        \end{aligned}

    .. math::
        \begin{aligned}
            \left [ \phi_2 | \{ x_0 \} \right]
                &= \frac{1}{4} e^\frac{1}{4} - \frac{1}{8} e^\frac{1}{2} \\
            \left [ \phi_2 | \{ x_1 \} \right] &= \frac{1}{8} e^\frac{1}{2} \\
            \left [ \phi_2 | \{ x_0, x_1 \} \right] &= \frac{1}{8} e^\frac{1}{2}
        \end{aligned}

    .. math:: \phi_2 = \frac{1}{12} e^\frac{1}{4} + \frac{1}{24} e^\frac{1}{2} \approx 0.1757

    Valuation of :math:`x_3`
    ========================

    .. math::
        \begin{aligned}
            \delta((x_2, x_3), \{ x_0 \}, 3) &= 0 &\quad
                \delta((x_3, x_2), \{ x_0 \}, 3) &= \frac{1}{4} e^\frac{1}{2} \\
            \delta((x_2, x_3), \{ x_1 \}, 3) &= 0 &\quad
                \delta((x_3, x_2), \{ x_1 \}, 3) &= \frac{1}{4} e^\frac{1}{2} \\
            \delta((x_2, x_3), \{ x_0, x_1 \}, 3) &= 0 &\quad
                \delta((x_3, x_2), \{ x_0, x_1 \}, 3) &= \frac{1}{4} e^\frac{1}{2}
        \end{aligned}

    .. math::
        \begin{aligned}
            \left [ \phi_3 | \{ x_0 \} \right] &= \frac{1}{8} e^\frac{1}{2} \\
            \left [ \phi_3 | \{ x_1 \} \right] &= \frac{1}{8} e^\frac{1}{2} \\
            \left [ \phi_3 | \{ x_0, x_1 \} \right] &= \frac{1}{8} e^\frac{1}{2}
        \end{aligned}

    .. math:: \phi_3 = \frac{1}{8} e^\frac{1}{2} \approx 0.2061
    """
    return (
        {
            "normalize_values": False,
        },
        ValuationResult(
            values=np.array(
                [
                    1 / 6 * np.exp(1 / 4),
                    1 / 3 * np.exp(1 / 4),
                    1 / 12 * np.exp(1 / 4) + 1 / 24 * np.exp(1 / 2),
                    1 / 8 * np.exp(1 / 2),
                ]
            )
        ),
        {"atol": 0.05},
    )


@pytest.fixture(scope="function")
def linear_classifier_cs_scorer_args_exact_solution_use_default_score_norm(
    linear_classifier_cs_scorer_args_exact_solution_use_default_score: Tuple[
        Dict, ValuationResult, Dict
    ]
) -> Tuple[Dict, ValuationResult, Dict]:
    """
    Same as :func:`linear_classifier_cs_scorer_args_exact_solution_use_default_score`
    but with normalization. The values of label c are normalized by the in-class score
    of label c divided by the sum of values of that specific label.
    """
    values = linear_classifier_cs_scorer_args_exact_solution_use_default_score[1].values
    label_zero_coefficient = 1 / np.exp(1 / 4)
    label_one_coefficient = 1 / (1 / 3 * np.exp(1 / 4) + 2 / 3 * np.exp(1 / 2))

    return (
        {
            "normalize_values": True,
        },
        ValuationResult(
            values=np.array(
                [
                    values[0] * label_zero_coefficient,
                    values[1] * label_zero_coefficient,
                    values[2] * label_one_coefficient,
                    values[3] * label_one_coefficient,
                ]
            )
        ),
        {"atol": 0.05},
    )


@pytest.fixture(scope="function")
def linear_classifier_cs_scorer_args_exact_solution_use_add_idx() -> Tuple[
    Dict, ValuationResult, Dict
]:
    r"""
    Returns the exact solution for the class wise shapley value of the training and
    validation set of the `utility_alt_seq_cf_linear_classifier_cs_scorer` fixture.

    ===========================
    CS-Shapley Manual Derivation
    ===========================

    :Author: Markus Semmler
    :Date:   August 2023

    Dataset description
    ===================

    We have a training and a test dataset. We want to model a simple XOR dataset. The
    development set :math:`D` is given by

    .. math::
        \begin{aligned}
            \hat{x}_0 &= 1 \quad &\hat{y}_0 = 0 \\
            \hat{x}_1 &= 2 \quad &\hat{y}_1 = 0 \\
            \hat{x}_2 &= 3 \quad &\hat{y}_2 = 0 \\
            \hat{x}_3 &= 4 \quad &\hat{y}_3 = 1 \\
        \end{aligned}

    and the training set :math:`T` is given by

    .. math::
        \begin{aligned}
            x_0 &= 1 \quad &y_0 = 0 \\
            x_1 &= 2 \quad &y_1 = 0 \\
            x_2 &= 3 \quad &y_2 = 1 \\
            x_3 &= 4 \quad &y_3 = 1 \\
        \end{aligned}

    Note that the training set and the development set contain the same
    inputs x, but differ in the label :math:`\hat{y}_2 \neq y_2`

    Model
    =====

    We use an adapted version of linear regression

    .. math:: y = \max(0, \min(1, \text{round}(\beta^T x)))

    for classification, with the closed form solution

    .. math:: \beta = \frac{\text{dot}(x, y)}{\text{dot}(x, x)}

    Fitted model
    ============

    The hyperparameters for all combinations are

    .. container:: tabular

       | \|c||Sc \| Sc \| Sc \| Sc \| :math:`S_1 \cup S_2` &
         :math:`\emptyset` & :math:`\{x_2\}` & :math:`\{x_3\}` &
         :math:`\{x_2, x_3\}`
       | :math:`\emptyset` & nan & :math:`\frac{1}{3}` & :math:`\frac{1}{4}`
         & :math:`\frac{7}{25}`
       | :math:`\{x_0\}` & :math:`0` & :math:`\frac{3}{10}` &
         :math:`\frac{4}{17}` & :math:`\frac{7}{26}`
       | :math:`\{x_1\}` & :math:`0` & :math:`\frac{3}{13}` &
         :math:`\frac{1}{5}` &\ :math:`\frac{7}{29}`
       | :math:`\{x_0, x_1 \}` & :math:`0` & :math:`\frac{3}{14}` &
         :math:`\frac{4}{21}` & :math:`\frac{7}{30}`

    Accuracy tables on development set :math:`D`
    ============================================

    .. container:: tabular

       | \|c||Sc \| Sc \| Sc \| Sc \| :math:`S_1 \cup S_2` &
         :math:`\emptyset` & :math:`\{x_2\}` & :math:`\{x_3\}` &
         :math:`\{x_2, x_3\}`
       | :math:`\emptyset` & :math:`0` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{4}` & :math:`\frac{1}{4}`
       | :math:`\{x_0\}` & :math:`\frac{3}{4}` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{2}` & :math:`\frac{1}{4}`
       | :math:`\{x_1\}` & :math:`\frac{3}{4}` & :math:`\frac{1}{2}` &
         :math:`\frac{1}{2}` &\ :math:`\frac{1}{2}`
       | :math:`\{x_0, x_1 \}` & :math:`\frac{3}{4}` & :math:`\frac{1}{2}` &
         :math:`\frac{1}{2}` & :math:`\frac{1}{2}`

    .. container:: tabular

       | \|c||Sc \| Sc \| Sc \| Sc \| :math:`S_1 \cup S_2` &
         :math:`\emptyset` & :math:`\{x_2\}` & :math:`\{x_3\}` &
         :math:`\{x_2, x_3\}`
       | :math:`\emptyset` & :math:`0` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{4}` & :math:`\frac{1}{4}`
       | :math:`\{x_0\}` & :math:`0` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{4}` & :math:`\frac{1}{4}`
       | :math:`\{x_1\}` & :math:`0` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{4}` &\ :math:`\frac{1}{4}`
       | :math:`\{x_0, x_1 \}` & :math:`0` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{4}` & :math:`\frac{1}{4}`

    CS-Shapley
    ==========

    The formulas of the algorithm are given by

    .. math::

        \begin{aligned}
            \delta(\pi, S_{-y_i}, i) &= v_{y_i}(\pi_{:i} \cup \{ i \} | S_{-y_i})
                - v_{y_i}(\pi_{:i} | S_{-y_i}) \\
            \left [ \phi_i | S_{-y_i} \right ] &= \frac{1}{|T_{y_i}|!}
                \sum_{\pi \in \Pi(T_{y_i})} \delta(\pi, S_{-y_i}, i) \\
            \phi_i &= \frac{1}{2^{|T_{-y_i}|}-1} \left [\sum_{\emptyset \subset S_{-y_i}
                \subseteq T_{-y_i}} \left [ \phi_i | S_{-y_i} \right ] \right ]
        \end{aligned}

    Valuation of :math:`x_0`
    ========================

    .. math::
        \begin{aligned}
            \delta((x_0, x_1), \{ x_2 \}, 0) &= 0 &\quad
                \delta((x_1, x_0), \{ x_2 \}, 0) &= 0 \\
            \delta((x_0, x_1), \{ x_3 \}, 0) &= \frac{1}{4} e^\frac{1}{4} &\quad
                \delta((x_1, x_0), \{ x_3 \}, 0) &= 0 \\
            \delta((x_0, x_1), \{ x_2, x_3 \}, 0) &= 0 &\quad
                \delta((x_1, x_0), \{ x_2, x_3 \}, 0) &= 0
        \end{aligned}

    .. math::
        \begin{aligned}
            \left [ \phi_0 | \{ x_2 \} \right] &= 0 \\
            \left [ \phi_0 | \{ x_3 \} \right] &= \frac{1}{8} e^\frac{1}{4} \\
            \left [ \phi_0 | \{ x_2, x_3 \} \right] &= 0
        \end{aligned}

    .. math:: \phi_0 = \frac{1}{24} e^\frac{1}{4} \approx 0.0535

    Valuation of :math:`x_1`
    ========================

    .. math::
        \begin{aligned}
            \delta((x_0, x_1), \{ x_2 \}, 1) &= \frac{1}{4} e^\frac{1}{4} &\quad
                \delta((x_1, x_0), \{ x_2 \}, 1) &= \frac{1}{4} e^\frac{1}{4} \\
            \delta((x_0, x_1), \{ x_3 \}, 1) &= 0 &\quad
                \delta((x_1, x_0), \{ x_3 \}, 1) &= \frac{1}{4} e^\frac{1}{4} \\
            \delta((x_0, x_1), \{ x_2, x_3 \}, 1) &= \frac{1}{4} e^\frac{1}{4} &\quad
                \delta((x_1, x_0), \{ x_2, x_3 \}, 1) &= \frac{1}{4} e^\frac{1}{4}
        \end{aligned}

    .. math::
        \begin{aligned}
            \left [ \phi_1 | \{ x_2 \} \right] &= \frac{1}{4} e^\frac{1}{4} \\
            \left [ \phi_1 | \{ x_3 \} \right] &= \frac{1}{8} e^\frac{1}{4} \\
            \left [ \phi_1 | \{ x_2, x_3 \} \right] &= \frac{1}{4} e^\frac{1}{4}
        \end{aligned}

    .. math:: \phi_0 = \frac{5}{24} e^\frac{1}{4} \approx 0.2675

    Valuation of :math:`x_2`
    ========================

    .. math::
        \begin{aligned}
            \delta((x_2, x_3), \{ x_0 \}, 2) &= \frac{1}{4} e^\frac{1}{4} &\quad
                \delta((x_3, x_2), \{ x_0 \}, 2)
                &= \frac{1}{4} e^\frac{1}{4} - \frac{1}{4} e^\frac{1}{2} \\
            \delta((x_2, x_3), \{ x_1 \}, 2) &= \frac{1}{4} e^\frac{1}{2} &\quad
                \delta((x_3, x_2), \{ x_1 \}, 2) &= 0 \\
            \delta((x_2, x_3), \{ x_0, x_1 \}, 2) &= \frac{1}{4} e^\frac{1}{2} &\quad
                \delta((x_3, x_2), \{ x_0, x_1 \}, 2) &= 0
        \end{aligned}

    .. math::
        \begin{aligned}
            \left [ \phi_2 | \{ x_0 \} \right]
                &= \frac{1}{4} e^\frac{1}{4} - \frac{1}{8} e^\frac{1}{2} \\
            \left [ \phi_2 | \{ x_1 \} \right] &= \frac{1}{8} e^\frac{1}{2} \\
            \left [ \phi_2 | \{ x_0, x_1 \} \right] &= \frac{1}{8} e^\frac{1}{2}
        \end{aligned}

    .. math:: \phi_2 = \frac{1}{12} e^\frac{1}{4} + \frac{1}{24} e^\frac{1}{2} \approx 0.1757

    Valuation of :math:`x_3`
    ========================

    .. math::
        \begin{aligned}
            \delta((x_2, x_3), \{ x_0 \}, 3) &= 0 &\quad
                \delta((x_3, x_2), \{ x_0 \}, 3) &= \frac{1}{4} e^\frac{1}{2} \\
            \delta((x_2, x_3), \{ x_1 \}, 3) &= 0 &\quad
                \delta((x_3, x_2), \{ x_1 \}, 3) &= \frac{1}{4} e^\frac{1}{2} \\
            \delta((x_2, x_3), \{ x_0, x_1 \}, 3) &= 0 &\quad
                \delta((x_3, x_2), \{ x_0, x_1 \}, 3) &= \frac{1}{4} e^\frac{1}{2}
        \end{aligned}

    .. math::
        \begin{aligned}
            \left [ \phi_3 | \{ x_0 \} \right] &= \frac{1}{8} e^\frac{1}{2} \\
            \left [ \phi_3 | \{ x_1 \} \right] &= \frac{1}{8} e^\frac{1}{2} \\
            \left [ \phi_3 | \{ x_0, x_1 \} \right] &= \frac{1}{8} e^\frac{1}{2}
        \end{aligned}

    .. math:: \phi_3 = \frac{1}{8} e^\frac{1}{2} \approx 0.2061
    """
    return (
        {
            "use_default_scorer_value": False,
            "normalize_values": False,
        },
        ValuationResult(
            values=np.array(
                [
                    1 / 24 * np.exp(1 / 4),
                    5 / 24 * np.exp(1 / 4),
                    1 / 12 * np.exp(1 / 4) + 1 / 24 * np.exp(1 / 2),
                    1 / 8 * np.exp(1 / 2),
                ]
            )
        ),
        {"atol": 0.05},
    )


@pytest.fixture(scope="function")
def linear_classifier_cs_scorer_args_exact_solution_use_add_idx_empty_set() -> Tuple[
    Dict, ValuationResult, Dict
]:
    r"""
    Returns the exact solution for the class wise shapley value of the training and
    validation set of the `utility_alt_seq_cf_linear_classifier_cs_scorer` fixture.

    ===========================
    CS-Shapley Manual Derivation
    ===========================

    :Author: Markus Semmler
    :Date:   August 2023

    Dataset description
    ===================

    We have a training and a test dataset. We want to model a simple XOR dataset. The
    development set :math:`D` is given by

    .. math::
        \begin{aligned}
            \hat{x}_0 &= 1 \quad &\hat{y}_0 = 0 \\
            \hat{x}_1 &= 2 \quad &\hat{y}_1 = 0 \\
            \hat{x}_2 &= 3 \quad &\hat{y}_2 = 0 \\
            \hat{x}_3 &= 4 \quad &\hat{y}_3 = 1 \\
        \end{aligned}

    and the training set :math:`T` is given by

    .. math::
        \begin{aligned}
            x_0 &= 1 \quad &y_0 = 0 \\
            x_1 &= 2 \quad &y_1 = 0 \\
            x_2 &= 3 \quad &y_2 = 1 \\
            x_3 &= 4 \quad &y_3 = 1 \\
        \end{aligned}

    Note that the training set and the development set contain the same
    inputs x, but differ in the label :math:`\hat{y}_2 \neq y_2`

    Model
    =====

    We use an adapted version of linear regression

    .. math:: y = \max(0, \min(1, \text{round}(\beta^T x)))

    for classification, with the closed form solution

    .. math:: \beta = \frac{\text{dot}(x, y)}{\text{dot}(x, x)}

    Fitted model
    ============

    The hyperparameters for all combinations are

    .. container:: tabular

       | \|c||Sc \| Sc \| Sc \| Sc \| :math:`S_1 \cup S_2` &
         :math:`\emptyset` & :math:`\{x_2\}` & :math:`\{x_3\}` &
         :math:`\{x_2, x_3\}`
       | :math:`\emptyset` & nan & :math:`\frac{1}{3}` & :math:`\frac{1}{4}`
         & :math:`\frac{7}{25}`
       | :math:`\{x_0\}` & :math:`0` & :math:`\frac{3}{10}` &
         :math:`\frac{4}{17}` & :math:`\frac{7}{26}`
       | :math:`\{x_1\}` & :math:`0` & :math:`\frac{3}{13}` &
         :math:`\frac{1}{5}` &\ :math:`\frac{7}{29}`
       | :math:`\{x_0, x_1 \}` & :math:`0` & :math:`\frac{3}{14}` &
         :math:`\frac{4}{21}` & :math:`\frac{7}{30}`

    Accuracy tables on development set :math:`D`
    ============================================

    .. container:: tabular

       | \|c||Sc \| Sc \| Sc \| Sc \| :math:`S_1 \cup S_2` &
         :math:`\emptyset` & :math:`\{x_2\}` & :math:`\{x_3\}` &
         :math:`\{x_2, x_3\}`
       | :math:`\emptyset` & :math:`0` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{4}` & :math:`\frac{1}{4}`
       | :math:`\{x_0\}` & :math:`\frac{3}{4}` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{2}` & :math:`\frac{1}{4}`
       | :math:`\{x_1\}` & :math:`\frac{3}{4}` & :math:`\frac{1}{2}` &
         :math:`\frac{1}{2}` &\ :math:`\frac{1}{2}`
       | :math:`\{x_0, x_1 \}` & :math:`\frac{3}{4}` & :math:`\frac{1}{2}` &
         :math:`\frac{1}{2}` & :math:`\frac{1}{2}`

    .. container:: tabular

       | \|c||Sc \| Sc \| Sc \| Sc \| :math:`S_1 \cup S_2` &
         :math:`\emptyset` & :math:`\{x_2\}` & :math:`\{x_3\}` &
         :math:`\{x_2, x_3\}`
       | :math:`\emptyset` & :math:`0` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{4}` & :math:`\frac{1}{4}`
       | :math:`\{x_0\}` & :math:`0` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{4}` & :math:`\frac{1}{4}`
       | :math:`\{x_1\}` & :math:`0` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{4}` &\ :math:`\frac{1}{4}`
       | :math:`\{x_0, x_1 \}` & :math:`0` & :math:`\frac{1}{4}` &
         :math:`\frac{1}{4}` & :math:`\frac{1}{4}`

    CS-Shapley
    ==========

    The formulas of the algorithm are given by

    .. math::

        \begin{aligned}
            \delta(\pi, S_{-y_i}, i) &= v_{y_i}(\pi_{:i} \cup \{ i \} | S_{-y_i})
                - v_{y_i}(\pi_{:i} | S_{-y_i}) \\
            \left [ \phi_i | S_{-y_i} \right ] &= \frac{1}{|T_{y_i}|!}
                \sum_{\pi \in \Pi(T_{y_i})} \delta(\pi, S_{-y_i}, i) \\
            \phi_i &= \frac{1}{2^{|T_{-y_i}|}} \left [\sum_{S_{-y_i}
                \subseteq T_{-y_i}} \left [ \phi_i | S_{-y_i} \right ] \right ]
        \end{aligned}

    Valuation of :math:`x_0`
    ========================

    .. math::
        \begin{aligned}
            \delta((x_0, x_1), \emptyset, 0) &= \frac{3}{4} &\quad
                \delta((x_1, x_0), \emptyset, 0) &= 0 \\
            \delta((x_0, x_1), \{ x_2 \}, 0) &= 0 &\quad
                \delta((x_1, x_0), \{ x_2 \}, 0) &= 0 \\
            \delta((x_0, x_1), \{ x_3 \}, 0) &= \frac{1}{4} e^\frac{1}{4} &\quad
                \delta((x_1, x_0), \{ x_3 \}, 0) &= 0 \\
            \delta((x_0, x_1), \{ x_2, x_3 \}, 0) &= 0 &\quad
                \delta((x_1, x_0), \{ x_2, x_3 \}, 0) &= 0
        \end{aligned}

    .. math::
        \begin{aligned}
            \left [ \phi_0 | \emptyset \right] &= \frac{3}{8} \\
            \left [ \phi_0 | \{ x_2 \} \right] &= 0 \\
            \left [ \phi_0 | \{ x_3 \} \right] &= \frac{1}{8} e^\frac{1}{4} \\
            \left [ \phi_0 | \{ x_2, x_3 \} \right] &= 0
        \end{aligned}

    .. math:: \phi_0 = \frac{3}{32} + \frac{1}{32} e^\frac{1}{4} \approx 0.1339

    Valuation of :math:`x_1`
    ========================

    .. math::
        \begin{aligned}
            \delta((x_0, x_1), \emptyset, 1) &= 0 &\quad
                \delta((x_1, x_0), \emptyset, 1) &= \frac{3}{4} \\
            \delta((x_0, x_1), \{ x_2 \}, 1) &= \frac{1}{4} e^\frac{1}{4} &\quad
                \delta((x_1, x_0), \{ x_2 \}, 1) &= \frac{1}{4} e^\frac{1}{4} \\
            \delta((x_0, x_1), \{ x_3 \}, 1) &= 0 &\quad
                \delta((x_1, x_0), \{ x_3 \}, 1) &= \frac{1}{4} e^\frac{1}{4} \\
            \delta((x_0, x_1), \{ x_2, x_3 \}, 1) &= \frac{1}{4} e^\frac{1}{4} &\quad
                \delta((x_1, x_0), \{ x_2, x_3 \}, 1) &= \frac{1}{4} e^\frac{1}{4}
        \end{aligned}

    .. math::
        \begin{aligned}
            \left [ \phi_1 | \emptyset \right] &= \frac{3}{8} \\
            \left [ \phi_1 | \{ x_2 \} \right] &= \frac{1}{4} e^\frac{1}{4} \\
            \left [ \phi_1 | \{ x_3 \} \right] &= \frac{1}{8} e^\frac{1}{4} \\
            \left [ \phi_1 | \{ x_2, x_3 \} \right] &= \frac{1}{4} e^\frac{1}{4}
        \end{aligned}

    .. math:: \phi_0 = \frac{3}{32} + \frac{5}{32} e^\frac{1}{4} \approx 0.2944

    Valuation of :math:`x_2`
    ========================

    .. math::
        \begin{aligned}
            \delta((x_2, x_3), \emptyset, 2) &= \frac{1}{4} e^\frac{1}{4} &\quad
                \delta((x_3, x_2), \emptyset, 2) &= 0 \\
            \delta((x_2, x_3), \{ x_0 \}, 2) &= \frac{1}{4} e^\frac{1}{4} &\quad
                \delta((x_3, x_2), \{ x_0 \}, 2)
                &= \frac{1}{4} e^\frac{1}{4} - \frac{1}{4} e^\frac{1}{2} \\
            \delta((x_2, x_3), \{ x_1 \}, 2) &= \frac{1}{4} e^\frac{1}{2} &\quad
                \delta((x_3, x_2), \{ x_1 \}, 2) &= 0 \\
            \delta((x_2, x_3), \{ x_0, x_1 \}, 2) &= \frac{1}{4} e^\frac{1}{2} &\quad
                \delta((x_3, x_2), \{ x_0, x_1 \}, 2) &= 0
        \end{aligned}

    .. math::
        \begin{aligned}
            \left [ \phi_2 | \emptyset \right] &= \frac{1}{8} e^\frac{1}{4} \\
            \left [ \phi_2 | \{ x_0 \} \right]
                &= \frac{1}{4} e^\frac{1}{4} - \frac{1}{8} e^\frac{1}{2} \\
            \left [ \phi_2 | \{ x_1 \} \right] &= \frac{1}{8} e^\frac{1}{2} \\
            \left [ \phi_2 | \{ x_0, x_1 \} \right] &= \frac{1}{8} e^\frac{1}{2}
        \end{aligned}

    .. math::
        \phi_2 = \frac{5}{32} e^\frac{1}{4} + \frac{1}{32} e^\frac{1}{2} \approx 0.2522

    Valuation of :math:`x_3`
    ========================

    .. math::
        \begin{aligned}
            \delta((x_2, x_3), \emptyset, 3) &= 0 &\quad
                \delta((x_3, x_2), \emptyset, 3) &= \frac{1}{4} e^\frac{1}{4} \\
            \delta((x_2, x_3), \{ x_0 \}, 3) &= 0 &\quad
                \delta((x_3, x_2), \{ x_0 \}, 3) &= \frac{1}{4} e^\frac{1}{2} \\
            \delta((x_2, x_3), \{ x_1 \}, 3) &= 0 &\quad
                \delta((x_3, x_2), \{ x_1 \}, 3) &= \frac{1}{4} e^\frac{1}{2} \\
            \delta((x_2, x_3), \{ x_0, x_1 \}, 3) &= 0 &\quad
                \delta((x_3, x_2), \{ x_0, x_1 \}, 3) &= \frac{1}{4} e^\frac{1}{2}
        \end{aligned}

    .. math::
        \begin{aligned}
            \left [ \phi_3 | \emptyset \right] &= \frac{1}{8} e^\frac{1}{4} \\
            \left [ \phi_3 | \{ x_0 \} \right] &= \frac{1}{8} e^\frac{1}{2} \\
            \left [ \phi_3 | \{ x_1 \} \right] &= \frac{1}{8} e^\frac{1}{2} \\
            \left [ \phi_3 | \{ x_0, x_1 \} \right] &= \frac{1}{8} e^\frac{1}{2}
        \end{aligned}

    .. math::
        \phi_3 = \frac{1}{32} e^\frac{1}{4} + \frac{3}{32} e^\frac{1}{2} \approx 0.1947
    """
    return (
        {
            "use_default_scorer_value": False,
            "min_elements_per_label": 0,
            "normalize_values": False,
        },
        ValuationResult(
            values=np.array(
                [
                    3 / 32 + 1 / 32 * np.exp(1 / 4),
                    3 / 32 + 5 / 32 * np.exp(1 / 4),
                    5 / 32 * np.exp(1 / 4) + 1 / 32 * np.exp(1 / 2),
                    1 / 32 * np.exp(1 / 4) + 3 / 32 * np.exp(1 / 2),
                ]
            )
        ),
        {"atol": 0.05},
    )


@pytest.mark.parametrize("n_samples", [500], ids=lambda x: "n_samples={}".format(x))
@pytest.mark.parametrize(
    "n_resample_complement_sets",
    [1],
    ids=lambda x: "n_resample_complement_sets={}".format(x),
)
@pytest.mark.parametrize(
    "linear_classifier_cs_scorer_args_exact_solution",
    [
        "linear_classifier_cs_scorer_args_exact_solution_use_default_score",
        "linear_classifier_cs_scorer_args_exact_solution_use_default_score_norm",
        "linear_classifier_cs_scorer_args_exact_solution_use_add_idx",
        "linear_classifier_cs_scorer_args_exact_solution_use_add_idx_empty_set",
    ],
)
def test_classwise_shapley(
    linear_classifier_cs_scorer: Utility,
    linear_classifier_cs_scorer_args_exact_solution: Tuple[Dict, ValuationResult],
    n_samples: int,
    n_resample_complement_sets: int,
    request,
):
    args, exact_solution, check_args = request.getfixturevalue(
        linear_classifier_cs_scorer_args_exact_solution
    )
    values = compute_classwise_shapley_values(
        linear_classifier_cs_scorer,
        done=MaxChecks(n_samples),
        truncation=NoTruncation(),
        n_resample_complement_sets=n_resample_complement_sets,
        **args,
        progress=True,
    )
    check_values(values, exact_solution, **check_args)
    assert np.all(values.counts == n_samples * n_resample_complement_sets)


@pytest.mark.parametrize(
    "dataset_alt_seq_simple",
    [((101, 0.3, 0.4))],
    indirect=True,
)
def test_cs_scorer_on_dataset_alt_seq_simple(dataset_alt_seq_simple):
    """
    Tests the class wise scorer.
    """

    scorer = ClasswiseScorer("accuracy", initial_label=0)
    assert str(scorer) == "classwise accuracy"
    assert repr(scorer) == "ClasswiseAccuracy (scorer=make_scorer(accuracy_score))"

    x, y, info = dataset_alt_seq_simple
    n_element = len(x)
    target_in_cls_acc_0 = (info["left_margin"] * 100 + 1) / n_element
    target_out_of_cls_acc_0 = (info["right_margin"] * 100 + 1) / n_element

    model = ThresholdClassifier()
    in_cls_acc_0, out_of_cls_acc_0 = scorer.estimate_in_cls_and_out_of_cls_score(
        model, x, y
    )
    assert np.isclose(in_cls_acc_0, target_in_cls_acc_0)
    assert np.isclose(out_of_cls_acc_0, target_out_of_cls_acc_0)

    scorer.label = 1
    in_cls_acc_1, out_of_cls_acc_1 = scorer.estimate_in_cls_and_out_of_cls_score(
        model, x, y
    )
    assert in_cls_acc_1 == out_of_cls_acc_0
    assert in_cls_acc_0 == out_of_cls_acc_1

    scorer.label = 0
    value = scorer(model, x, y)
    assert np.isclose(value, in_cls_acc_0 * np.exp(out_of_cls_acc_0))

    scorer.label = 1
    value = scorer(model, x, y)
    assert np.isclose(value, in_cls_acc_1 * np.exp(out_of_cls_acc_1))


def test_cs_scorer_on_alt_seq_cf_linear_classifier_cs_score(
    linear_classifier_cs_scorer: Utility,
):
    subsets_zero = list(powerset(np.array((0, 1))))
    subsets_one = list(powerset(np.array((2, 3))))
    subsets_zero = [tuple(s) for s in subsets_zero]
    subsets_one = [tuple(s) for s in subsets_one]
    target_betas = pd.DataFrame(
        [
            [np.nan, 1 / 3, 1 / 4, 7 / 25],
            [0, 3 / 10, 4 / 17, 7 / 26],
            [0, 3 / 13, 1 / 5, 7 / 29],
            [0, 3 / 14, 4 / 21, 7 / 30],
        ],
        index=subsets_zero,
        columns=subsets_one,
    )
    target_accuracies_zero = pd.DataFrame(
        [
            [0, 1 / 4, 1 / 4, 1 / 4],
            [3 / 4, 1 / 4, 1 / 2, 1 / 4],
            [3 / 4, 1 / 2, 1 / 2, 1 / 2],
            [3 / 4, 1 / 2, 1 / 2, 1 / 2],
        ],
        index=subsets_zero,
        columns=subsets_one,
    )
    target_accuracies_one = pd.DataFrame(
        [
            [0, 1 / 4, 1 / 4, 1 / 4],
            [0, 1 / 4, 1 / 4, 1 / 4],
            [0, 1 / 4, 1 / 4, 1 / 4],
            [0, 1 / 4, 1 / 4, 1 / 4],
        ],
        index=subsets_zero,
        columns=subsets_one,
    )
    model = linear_classifier_cs_scorer.model
    scorer = cast(ClasswiseScorer, linear_classifier_cs_scorer.scorer)
    scorer.label = 0

    for set_zero_idx in range(len(subsets_zero)):
        for set_one_idx in range(len(subsets_one)):
            indices = list(subsets_zero[set_zero_idx] + subsets_one[set_one_idx])
            (
                x_train,
                y_train,
            ) = linear_classifier_cs_scorer.data.get_training_data(indices)
            linear_classifier_cs_scorer.model.fit(x_train, y_train)
            fitted_beta = linear_classifier_cs_scorer.model._beta  # noqa
            target_beta = target_betas.iloc[set_zero_idx, set_one_idx]
            assert (
                np.isnan(fitted_beta)
                if np.isnan(target_beta)
                else fitted_beta == target_beta
            )

            (
                x_test,
                y_test,
            ) = linear_classifier_cs_scorer.data.get_test_data()
            in_cls_acc_0, in_cls_acc_1 = scorer.estimate_in_cls_and_out_of_cls_score(
                model, x_test, y_test
            )
            assert (
                in_cls_acc_0 == target_accuracies_zero.iloc[set_zero_idx, set_one_idx]
            )
            assert in_cls_acc_1 == target_accuracies_one.iloc[set_zero_idx, set_one_idx]


class ThresholdClassifier:
    def fit(self, x: NDArray, y: NDArray) -> float:
        raise NotImplementedError("Mock model")

    def predict(self, x: NDArray) -> NDArray:
        y = 0.5 < x
        return y[:, 0].astype(int)

    def score(self, x: NDArray, y: NDArray) -> float:
        raise NotImplementedError("Mock model")


class ClosedFormLinearClassifier:
    def __init__(self):
        self._beta = None

    def fit(self, x: NDArray, y: NDArray) -> float:
        v = x[:, 0]
        self._beta = np.dot(v, y) / np.dot(v, v)
        return -1

    def predict(self, x: NDArray) -> NDArray:
        if self._beta is None:
            raise AttributeError("Model not fitted")

        x = x[:, 0]
        probs = self._beta * x
        return np.clip(np.round(probs + 1e-10), 0, 1).astype(int)

    def score(self, x: NDArray, y: NDArray) -> float:
        pred_y = self.predict(x)
        return np.sum(pred_y == y) / 4


@pytest.fixture(scope="function")
def linear_classifier_cs_scorer(
    dataset_alt_seq_full: Dataset,
) -> Utility:
    return Utility(
        ClosedFormLinearClassifier(),
        dataset_alt_seq_full,
        ClasswiseScorer("accuracy"),
        catch_errors=False,
    )


@pytest.fixture(scope="function")
def dataset_alt_seq_full() -> Dataset:
    x_train = np.arange(1, 5).reshape([-1, 1])
    y_train = np.array([0, 0, 1, 1])
    x_test = x_train
    y_test = np.array([0, 0, 0, 1])
    return Dataset(x_train, y_train, x_test, y_test)

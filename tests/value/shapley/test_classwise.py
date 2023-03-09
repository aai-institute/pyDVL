"""
Test cases for the class wise shapley value.
"""
import random
from random import seed
from typing import Dict, Tuple

import numpy as np
import pytest

from pydvl.utils import Utility
from pydvl.value import MaxChecks, ValuationResult
from pydvl.value.shapley.classwise import compute_classwise_shapley_values
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
def linear_classifier_cs_scorer_args_exact_solution_use_default_score_norm() -> Tuple[
    Dict, ValuationResult, Dict
]:
    label_one_coefficient = 1 / 3 * np.exp(1 / 4) + 2 / 3 * np.exp(1 / 2)
    return (
        {
            "normalize_values": True,
        },
        ValuationResult(
            values=np.array(
                [
                    1 / 6,
                    1 / 3,
                    (1 / 12 * np.exp(1 / 4) + 1 / 24 * np.exp(1 / 2))
                    / label_one_coefficient,
                    1 / 8 * np.exp(1 / 2) / label_one_coefficient,
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
        done=MaxChecks(n_samples - 1),
        truncation=NoTruncation(),
        n_resample_complement_sets=n_resample_complement_sets,
        **args,
        progress=True,
    )
    check_values(values, exact_solution, **check_args)
    assert np.all(values.counts == n_samples * n_resample_complement_sets)

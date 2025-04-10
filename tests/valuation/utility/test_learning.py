from typing import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

from pydvl.utils import SupervisedModel
from pydvl.valuation import DataUtilityLearning
from pydvl.valuation.types import Sample
from pydvl.valuation.utility.base import UtilityBase
from pydvl.valuation.utility.learning import IndicatorUtilityModel


class LinearUtility(UtilityBase):
    """A utility function that returns the sum of the weights corresponding to the
    indices in the subset.

     u = w_0 * x_0 + w_1 * x_1 + w_2 * x_2 + ...

     where x_i = 1 if i is in the sample, 0 otherwise.
    """

    def __init__(self, weights: Sequence, training_data: Sequence):
        self.weights = np.array(weights)

        # FIXME this doesn't make sense
        self._training_data = training_data

    def __call__(self, sample: Sample):
        # Compute the sum of the weights corresponding to the indices in the subset.
        if sample is None or len(sample.subset) == 0:
            return 0.0
        return float(sum(self.weights[i] for i in sample.subset))


@pytest.fixture
def dul_instance():
    """A simple linear function with three features. The training data determines the
    dimension (3). A LinearRegression model is used to learn the utility with a small
    budget of three samples before shifting to prediction."""

    weights = [1, 2, 3]
    training_data = [0, 1, 2]
    dummy_util = LinearUtility(weights, training_data)
    predictor = LinearRegression(fit_intercept=False)
    model = IndicatorUtilityModel(predictor=predictor, n_data=len(training_data))
    training_budget = 3

    return DataUtilityLearning(dummy_util, training_budget, model)


def test_training_phase(dul_instance):
    """Test that during the training phase (i.e. before the budget is reached),
    calls to the DUL instance use the underlying utility and record the samples."""
    dul = dul_instance

    sample1 = Sample(0, np.array([0]))
    sample2 = Sample(1, np.array([1]))
    sample3 = Sample(2, np.array([2]))

    # These calls should use the DummyUtility directly.
    assert dul(sample1) == dul.utility(sample1)
    assert dul(sample2) == dul.utility(sample2)
    assert dul(sample3) == dul.utility(sample3)

    # At this point the training budget is exactly reached.
    assert len(dul._utility_samples) == dul.training_budget
    assert not dul._is_fitted


def test_prediction_phase(dul_instance):
    """
    Once the training budget is reached, the next call (with a new sample) should
    trigger the fitting of the linear model. The prediction should closely match
    the true utility computed by DummyUtility.
    """
    dul = dul_instance
    # Fill the training budget
    for i in range(3):
        dul(Sample(i, np.array([i])))

    # Create a new sample that was not seen during training.
    # For subset (0,2), the true utility is 1 (from index 0) + 3 (from index 2) = 4.
    sample_new = Sample(4, np.array([0, 2]))
    pred = dul(sample_new)

    assert dul._is_fitted
    np.testing.assert_allclose(pred, 4.0, rtol=1e-8)


def test_forwarding_existing_sample(dul_instance):
    """
    Test that if a sample that was already used during training is provided after
    the model is fitted, the stored (true) utility is returned rather than using the model.
    """
    dul = dul_instance
    # Fill the training budget
    us = [dul(Sample(i, np.array([i]))) for i in range(3)]

    # Trigger the model fitting by providing a new sample.
    sample_new = Sample(0, np.array([0, 2]))
    _ = dul(sample_new)  # should be 4.0

    # Call with a sample that was used during training.
    # The value should match the stored true utility.
    val2_again = dul(Sample(2, np.array([2])))
    assert val2_again == us[2]


def test_empty_sample(dul_instance):
    """
    Test that if an empty sample (or None) is provided, the call is forwarded
    directly to the underlying utility.
    """
    empty_sample = Sample(1, np.empty(0, dtype=int))
    assert dul_instance(empty_sample) == 0.0


class FakePredictor(SupervisedModel):
    """A fake predictor that simply records the inputs to fit() and predict()
    and predicts the row-sum so that we can verify encoding indirectly."""

    def __init__(self):
        self.fit_X = None
        self.fit_y = None
        self.last_predict_X = None

    def fit(self, X, y):
        self.fit_X = X
        self.fit_y = y

    def predict(self, X):
        self.last_predict_X = X
        return np.sum(X, axis=1, keepdims=True)


@pytest.mark.parametrize(
    "utility_samples, encoding",
    [
        ({Sample(idx=0, subset=np.array([1, 3])): 2.5}, np.array([[0, 1, 0, 1, 0]])),
        ({Sample(idx=1, subset=np.array([0, 4])): 3.5}, np.array([[1, 0, 0, 0, 1]])),
        (
            {
                Sample(idx=0, subset=np.array([1, 3])): 2.5,
                Sample(idx=1, subset=np.array([0, 4])): 3.5,
            },
            np.array([[0, 1, 0, 1, 0], [1, 0, 0, 0, 1]]),
        ),
    ],
)
def test_indicator_utility_model_encoding(utility_samples, encoding: NDArray):
    """Verify that calling fit() encodes each sample correctly into a one-hot vector."""
    n_data = 5
    predictor = FakePredictor()
    model = IndicatorUtilityModel(predictor=predictor, n_data=n_data)

    model.fit(utility_samples)

    np.testing.assert_array_equal(predictor.fit_X, encoding)
    np.testing.assert_array_equal(
        predictor.fit_y, np.array(list(utility_samples.values())).reshape(-1, 1)
    )

    prediction = model.predict(list(utility_samples.keys()))
    expected_prediction = np.sum(encoding, axis=1).reshape(-1, 1)
    np.testing.assert_array_equal(predictor.last_predict_X, encoding)
    np.testing.assert_array_equal(prediction, expected_prediction)

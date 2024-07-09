"""Specialized scorer for k-nearest neighbors models."""


import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KNeighborsClassifier

from pydvl.valuation.scorers import SupervisedScorer


class KNNClassifierScorer(SupervisedScorer):
    """Scorer for KNN classifier models  based on the KNN likelihood.

    Typically, users will not create instances of this class directly but indirectly
    by using `pydvl.valuation.utility.KNNUtility`.

    Args:
        test_data: The test data to evaluate the model on.

    """

    def __init__(self, test_data):
        def scoring(model: KNeighborsClassifier, X: NDArray, y: NDArray) -> float:
            probs = model.predict_proba(X)
            label_to_pos = {label: i for i, label in enumerate(model.classes_)}
            likelihoods = []
            for i in range(len(y)):
                if y[i] not in label_to_pos:
                    likelihoods.append(0.0)
                else:
                    likelihoods.append(probs[i, label_to_pos[y[i]]])
            return np.mean(likelihoods)

        super().__init__(
            scoring=scoring,
            test_data=test_data,
            default=0.0,
            range=(0, 1),
            name="KNN Scorer",
        )

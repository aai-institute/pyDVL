import numpy as np
from numpy._typing import NDArray


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

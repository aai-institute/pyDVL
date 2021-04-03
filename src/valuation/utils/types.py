from abc import abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin


# Pedantic: only here for the type hints.
class Regressor(BaseEstimator, RegressorMixin):
    @abstractmethod
    def fit(self, x, y):
        pass

    pass

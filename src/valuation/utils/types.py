from abc import abstractmethod
from typing import Union

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


# Pedantic: only here for the type hints.
class Regressor(BaseEstimator, RegressorMixin):
    @abstractmethod
    def fit(self, x, y):
        pass

    pass


class Classifier(BaseEstimator, ClassifierMixin):
    @abstractmethod
    def fit(self, x, y):
        pass

    pass


SupervisedModel = Union[Regressor, Classifier]

__version__ = "0.1.0-dev1"

from abc import abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin

_logger = None


def set_logger(logger=None):
    global _logger
    if logger is not None:
        _logger = logger
    else:
        import logging
        _logger = logging.getLogger()


set_logger()


# Pedantic: only here for the type hints.
class Regressor(BaseEstimator, RegressorMixin):
    @abstractmethod
    def fit(self, x, y):
        pass

    pass

import numpy as np
import pandas as pd

from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split


class Dataset:
    """ Meh... Just a bunch of properties and shortcuts.
     I should probably ditch / redesign this. """
    def __init__(self,
                 data: Bunch,
                 train_size: float = 0.8,
                 random_state: int = None):
        x_train, x_test, y_train, y_test = \
            train_test_split(data.data, data.target,
                             train_size=train_size, random_state=random_state)
        try:
            self.target_names = data.target_names
        except AttributeError:
            pass

        self.x_train = pd.DataFrame(x_train, columns=data.feature_names)
        self.y_train = pd.DataFrame(y_train, columns=["target"])
        self.x_test = pd.DataFrame(x_test, columns=data.feature_names)
        self.y_test = pd.DataFrame(y_test, columns=["target"])
        self._description = data.DESCR
        self._locs = list(range(len(self.x_train)))

        assert np.all(self.x_train.index == self.y_train.index), "huh?"

    @property
    def ilocs(self):
        """ Contiguous integer index of positions in data.x_train.
        This is intended to be used with DataFrame.iloc[], as opposed to the
        values of data.x_train.index (labels) which are used with loc[]
        """
        return self._locs

    @property
    def dim(self):
        """ Returns the number of dimensions of a sample. """
        return self.x_train.shape[1]

    # hacky 🙈
    def __str__(self):
        return self._description

    def __len__(self):
        return len(self.x_train)

    @classmethod
    def from_sklearn_bunch(cls,
                           data: Bunch,
                           train_size: float = 0.8,
                           random_state: int = None) -> 'Dataset':
        raise NotImplementedError("In progress (need to change the constructor)")
        x_train, x_test, y_train, y_test = \
            train_test_split(data.data, data.target,
                             train_size=train_size, random_state=random_state)
        try:
            target_names = data.target_names
        except AttributeError:
            pass

        x_train = pd.DataFrame(x_train, columns=data.feature_names)
        y_train = pd.DataFrame(y_train, columns=["target"])
        x_test = pd.DataFrame(x_test, columns=data.feature_names)
        y_test = pd.DataFrame(y_test, columns=["target"])

        return Dataset(x_train, y_train, x_test, y_test, target_names,
                       data.DESCR)

    @classmethod
    def from_pandas_dataframe(cls, df: pd.DataFrame) -> 'Dataset':
        pass

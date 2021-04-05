import pandas as pd
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split


class Dataset:
    """ Meh... """
    def __init__(self, data: Bunch,
                 train_size: float = 0.8,
                 random_state: int = None):
        x_train, x_test, y_train, y_test = \
            train_test_split(data.data, data.target,
                             train_size=train_size, random_state=random_state)
        try:
            target_names = data.target_names
        except AttributeError:
            target_names = ['target']

        self.x_train = pd.DataFrame(x_train, columns=data.feature_names)
        self.y_train = pd.DataFrame(y_train, columns=target_names)
        self.x_test = pd.DataFrame(x_test, columns=data.feature_names)
        self.y_test = pd.DataFrame(y_test, columns=target_names)
        self._description = data.DESCR

        assert (self.x_train.index == self.y_train.index).all(), "huh?"

    @property
    def index(self):
        """ Shorthand for Dataset.x_train.index """
        # Ok, it might be confusing to have index == x_train.index...
        return self.x_train.index

    @property
    def dim(self):
        """ Returns the number of dimensions of a sample. """
        return self.x_train.shape[1]

    # hacky 🙈
    def __str__(self):
        return self._description

    def __len__(self):
        return len(self.x_train)

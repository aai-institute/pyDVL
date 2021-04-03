import pandas as pd
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split


class Dataset:
    """ Meh... """
    def __init__(self, data: Bunch, train_size: float = 0.8,
                 random_state: int = None):
        x_train, x_test, y_train, y_test = \
            train_test_split(data.data, data.target,
                             train_size=train_size, random_state=random_state)
        self.x_train = pd.DataFrame(x_train, columns=data.feature_names)
        self.y_train = pd.DataFrame(y_train, columns=['target'])
        self.x_test = pd.DataFrame(x_test, columns=data.feature_names)
        self.y_test = pd.DataFrame(y_test, columns=['target'])
        self._description = data.DESCR

    # hacky 🙈
    def __str__(self):
        return self._description

    def __len__(self):
        return len(self.x_train)

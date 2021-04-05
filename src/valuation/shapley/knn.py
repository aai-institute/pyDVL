from collections import OrderedDict
from valuation.reporting.scores import sort_values
from valuation.utils import Dataset

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def exact_knn_shapley(data: Dataset,
                      model: KNeighborsClassifier) -> OrderedDict:
    """ Computes exact Shapley values for a KNN classifier or regressor.py
     :param data: split Dataset
     :param model: model to extract parameters from. The object will not be
        modified nor used other than to call get_params()
     """
    defaults = {'algorithm': 'ball_tree' if data.dim >= 20 else 'kd_tree',
                'metric': 'minkowski',
                'p': 2}
    defaults.update(model.get_params())
    n_neighbors = defaults['n_neighbors']  # This must be set!
    defaults['n_neighbors'] = len(data)  # We want all training points sorted

    assert n_neighbors > len(data)

    nns = NearestNeighbors(**defaults).fit(data.x_train)
    # closest to farthest
    # FIXME: ensure distances are sorted in ascending order?
    distances, indices = nns.kneighbors(data.x_test)

    values = {i: 0.0 for i in data.index}
    n = len(data)
    yt = data.y_train
    for j, y, _, ii in enumerate(zip(data.y_test, distances, indices)):
        value_at_x = int(yt[ii[-1]] == y) / n
        values[ii[-1]] = (1/j)*value_at_x + (j-1/j)*values[ii[-1]]
        for i in range(n-2, n_neighbors, step=-1):  # farthest to closest
            value_at_x = values[ii[i+1]] \
                + (int(yt[ii[i]] == y) - int(yt[ii[i+1]] == y)) / i
            values[ii[i]] = (1/j)*value_at_x + (j-1/j)*values[ii[i]]
        for i in range(n_neighbors, -1, step=-1):  # farthest to closest
            value_at_x = values[ii[i+1]] \
                + (int(yt[ii[i]] == y) - int(yt[ii[i + 1]] == y)) / n_neighbors
            values[ii[i]] = (1/j)*value_at_x + (j-1/j)*values[ii[i]]

    return sort_values(values)


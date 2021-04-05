from collections import OrderedDict
from valuation.reporting.scores import sort_values
from valuation.utils import Dataset

from sklearn.neighbors import NearestNeighbors


def exact_knn_shapley(data: Dataset,
                      model_type: str,
                      **kwargs) -> OrderedDict:
    """ Computes exact Shapley values for a KNN classifier or regressor.py
     :param data: split Dataset
     :param model_type: either 'classifier' or 'regressor'. Sets the utility
        function to be the average prediction.
     :param kwargs: to be passed directly to `sklearn.neighbors.NearestNeighbors`
        to configure algorithm, metric, n_jobs, etc.
     """
    defaults = {'n_neighbors': len(data.x_test),
                'algorithm': 'ball_tree' if data.dim >= 20 else 'kd_tree',
                'metric': 'minkowski',
                'p': 2 }
    defaults.update(kwargs)
    nns = NearestNeighbors(**defaults).fit(data.x_train)
    distances, indices = nns.kneighbors(data.x_test)
    values = {i: 0.0 for i in data.index}
    for x, d, y in zip(data.x_test, distances, data.y_test[indices]):
        pass
    return sort_values(values)


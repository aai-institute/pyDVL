from sklearn.kernel_ridge import KernelRidge

from tests.conftest import check_rank_correlation
from valuation.loo.exact import linear_smoother_loo
from valuation.utils import Dataset, Utility


def test_linear_smoother_loo():
    from sklearn.datasets import load_boston
    data = Dataset.from_sklearn(load_boston())

    from sklearn.gaussian_process.kernels import RBF
    # model = GaussianProcessRegressor(kernel=RBF())
    model = KernelRidge(kernel='linear')
    model.fit(data.x_train, data.y_train)

    yhat = model.predict(data.x_test)

    from matplotlib import pyplot as plt
    plt.plot(data.y_test, label="true")
    plt.plot(yhat, label="pred")
    plt.legend()
    plt.show()

    values = linear_smoother_loo(model, data)

    from valuation.loo.naive import loo
    u = Utility(model, data, 'neg_mean_squared_error')
    naive = loo(u)

    check_rank_correlation(values, naive)

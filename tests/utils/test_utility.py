""" FIXME: more to do... """

from sklearn.linear_model import LinearRegression
from valuation.utils import Utility, powerset


# noinspection PyUnresolvedReferences
def test_cache(linear_dataset):
    u = Utility(model=LinearRegression(), data=linear_dataset, scoring="r2")
    subsets = list(powerset(u.data.indices))

    for s in subsets:
        u(s)
    assert u._utility_wrapper.cache_info().hits == 0

    for s in subsets:
        u(s)
    assert u._utility_wrapper.cache_info().hits == len(subsets)

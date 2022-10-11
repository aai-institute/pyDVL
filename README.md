<p align="center" style="text-align:center;">
    <img alt="pyDVL Logo" src="https://raw.githubusercontent.com/appliedAI-Initiative/pyDVL/develop/logo.svg" width="200"/>
</p>

<p align="center" style="text-align:center;">
    A library for data valuation.
</p>

<p align="center" style="text-align:center;">
    <a href="https://github.com/appliedAI-Initiative/pyDVL/actions/workflows/tox.yaml">
        <img src="https://github.com/appliedAI-Initiative/pyDVL/actions/workflows/tox.yaml/badge.svg" alt="Build Status"/>
    </a>
</p>

<p align="center" style="text-align:center;">
    <strong>
    <a href="https://appliedAI-Initiative.github.io/pyDVL">Docs</a>
    </strong>
</p>

pyDVL collects algorithms for Data Valuation and Influence Function computation.

Data Valuation is the task of estimating the intrinsic value of a data point
wrt. the training set, the model and a scoring function. We currently implement
methods from the following papers:

- Ghorbani, Amirata, and James Zou. ‘Data Shapley: Equitable Valuation of Data for
  Machine Learning’. In International Conference on Machine Learning, 2242–51.
  PMLR, 2019. http://proceedings.mlr.press/v97/ghorbani19c.html.
- Wang, Tianhao, Yu Yang, and Ruoxi Jia. ‘Improving Cooperative Game Theory-Based
  Data Valuation via Data Utility Learning’. arXiv, 2022.
  https://doi.org/10.48550/arXiv.2107.06336.
- Jia, Ruoxi, David Dao, Boxin Wang, Frances Ann Hubis, Nezihe Merve Gurel, Bo Li,
  Ce Zhang, Costas Spanos, and Dawn Song. ‘Efficient Task-Specific Data Valuation
  for Nearest Neighbor Algorithms’. Proceedings of the VLDB Endowment 12, no. 11 (1
  July 2019): 1610–23. https://doi.org/10.14778/3342263.3342637.

Influence Functions compute the effect that single points have on an estimator /
model. We implement methods from the following papers:

- Koh, Pang Wei, and Percy Liang. ‘Understanding Black-Box Predictions via
  Influence Functions’. In Proceedings of the 34th International Conference on
  Machine Learning, 70:1885–94. Sydney, Australia: PMLR, 2017.
  http://proceedings.mlr.press/v70/koh17a.html.

# Installation

To install the latest release use:

```shell
$ pip install pyDVL
```

You can also install the latest development version from
[TestPyPI](https://test.pypi.org/project/pyDVL/):

```shell
pip install pyDVL --index-url https://test.pypi.org/simple/
```

For more instructions and information refer to [Installing pyDVL
](https://appliedAI-Initiative.github.io/pyDVL/install.html) in the
documentation.

# Usage

pyDVL uses [Memcached](https://memcached.org/) to cache certain results and
speed up computation. You can run it either locally or, using
[Docker](https://www.docker.com/):

```shell
docker container run --rm -p 11211:11211 --name pydvl-cache -d memcached:latest
```

You can read more in the [caching module's
documentation](https://appliedAI-Initiative.github.io/pyDVL/pydvl/utils/caching.html).

Once that's done, the steps required to compute values for your samples are

1. Create a `Dataset` object with your train and test splits.
2. Create an instance of a `SupervisedModel` (basically any sklearn compatible
   predictor)
3. Create a `Utility` object to wrap the Dataset, the model and a scoring
   function.
4. Use one of the methods defined in the library to compute the values.

This is how it looks for *Truncated Montecarlo Shapley*, an efficient method for
Data Shapley values:

```python
import numpy as np
from pydvl.utils import Dataset, Utility
from pydvl.shapley.montecarlo import truncated_montecarlo_shapley
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = np.arange(100).reshape((50, 2)), np.arange(50)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=16
)
dataset = Dataset(X_train, X_test, y_train, y_test)
model = LinearRegression()
utility = Utility(model, dataset)
values, errors = truncated_montecarlo_shapley(u=utility, max_iterations=100)
```

For more instructions and information refer to [Getting
Started](https://appliedAI-Initiative.github.io/pyDVL/getting-started.html) in
the documentation. We provide several
[examples](https://appliedAI-Initiative.github.io/pyDVL/examples/index.html)
with details on the algorithms and their applications.

# Contributing

Please open new issues for bugs, feature requests and extensions. You can read
about the structure of the project, the toolchain and workflow in the [guide for
contributions](CONTRIBUTING.md).

# License

pyDVL is distributed under
[LGPL-3.0](https://www.gnu.org/licenses/lgpl-3.0.html). A complete version can
be found in two files: [here](LICENSE) and [here](COPYING.LESSER).

All contributions will be distributed under this license.

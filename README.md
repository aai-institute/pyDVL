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
    <br>
    <a href="https://pypi.org/project/pydvl/">
        <img src="https://img.shields.io/pypi/v/pydvl.svg"/>
    </a>
    <a href="https://pypi.org/project/pydvl/">
        <img src="https://img.shields.io/pypi/pyversions/pydvl.svg"/>
    </a>
    <img alt="PyPI - License" src="https://img.shields.io/pypi/l/pydvl"/>
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

- Castro, Javier, Daniel Gómez, and Juan Tejada. [Polynomial Calculation of the
  Shapley Value Based on Sampling](https://doi.org/10.1016/j.cor.2008.04.004).
  Computers & Operations Research, Selected papers presented at the Tenth
  International Symposium on Locational Decisions (ISOLDE X), 36, no. 5 (May 1,
  2009): 1726–30.
- Ghorbani, Amirata, and James Zou. [Data Shapley: Equitable Valuation of Data
  for Machine Learning](http://proceedings.mlr.press/v97/ghorbani19c.html). In
  International Conference on Machine Learning, 2242–51. PMLR, 2019.
- Wang, Tianhao, Yu Yang, and Ruoxi Jia. 
  [Improving Cooperative Game Theory-Based Data Valuation via Data Utility
  Learning](https://doi.org/10.48550/arXiv.2107.06336). arXiv, 2022.
- Jia, Ruoxi, David Dao, Boxin Wang, Frances Ann Hubis, Nezihe Merve Gurel, Bo
  Li, Ce Zhang, Costas Spanos, and Dawn Song. [Efficient Task-Specific Data
  Valuation for Nearest Neighbor Algorithms](https://doi.org/10.14778/3342263.3342637).
  Proceedings of the VLDB Endowment 12, no. 11 (1 July 2019): 1610–23.
- Okhrati, Ramin, and Aldo Lipani. [A Multilinear Sampling Algorithm to Estimate
  Shapley Values](https://doi.org/10.1109/ICPR48806.2021.9412511). In 25th
  International Conference on Pattern Recognition (ICPR 2020), 7992–99. IEEE,
  2021.
- Yan, T., & Procaccia, A. D. [If You Like Shapley Then You’ll Love the
  Core](https://ojs.aaai.org/index.php/AAAI/article/view/16721). Proceedings of
  the AAAI Conference on Artificial Intelligence, 35(6) (2021): 5751-5759.
- Jia, Ruoxi, David Dao, Boxin Wang, Frances Ann Hubis, Nick Hynes, Nezihe Merve
  Gürel, Bo Li, Ce Zhang, Dawn Song, and Costas J. Spanos. [Towards Efficient
  Data Valuation Based on the Shapley Value](http://proceedings.mlr.press/v89/jia19a.html).
  In 22nd International Conference on Artificial Intelligence and Statistics,
  1167–76. PMLR, 2019.
- Wang, Jiachen T., and Ruoxi Jia. [Data Banzhaf: A Robust Data Valuation
  Framework for Machine Learning](https://doi.org/10.48550/arXiv.2205.15466).
  arXiv, October 22, 2022.
- Kwon, Yongchan, and James Zou. [Beta Shapley: A Unified and Noise-Reduced Data
  Valuation Framework for Machine Learning](http://arxiv.org/abs/2110.14049).
  In Proceedings of the 25th International Conference on Artificial Intelligence
  and Statistics (AISTATS) 2022, Vol. 151. Valencia, Spain: PMLR, 2022.

Influence Functions compute the effect that single points have on an estimator /
model. We implement methods from the following papers:

- Koh, Pang Wei, and Percy Liang. [Understanding Black-Box Predictions via
  Influence Functions](http://proceedings.mlr.press/v70/koh17a.html). In
  Proceedings of the 34th International Conference on Machine Learning,
  70:1885–94. Sydney, Australia: PMLR, 2017.
- Naman Agarwal, Brian Bullins, and Elad Hazan, [Second-Order Stochastic Optimization
  for Machine Learning in Linear Time](https://www.jmlr.org/papers/v18/16-491.html),
  Journal of Machine Learning Research 18 (2017): 1-40.

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
](https://appliedAI-Initiative.github.io/pyDVL/20-install.html) in the
documentation.

# Usage

### Influence Functions

For influence computation, follow these steps:

1. Wrap your model and loss in a `TorchTwiceDifferential` object
2. Compute influence factors by providing training data and inversion method

Using the conjugate gradient algorithm, this would look like:
```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pydvl.influence import TorchTwiceDifferentiable, compute_influences, InversionMethod

nn_architecture = nn.Sequential(
    nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3),
    nn.Flatten(),
    nn.Linear(27, 3),
)
loss = nn.MSELoss()
model = TorchTwiceDifferentiable(nn_architecture, loss)

input_dim = (5, 5, 5)
output_dim = 3

train_data_loader = DataLoader(
    TensorDataset(torch.rand((10, *input_dim)), torch.rand((10, output_dim))),
    batch_size=2,
)
test_data_loader = DataLoader(
    TensorDataset(torch.rand((5, *input_dim)), torch.rand((5, output_dim))),
    batch_size=1,
)

influences = compute_influences(
    model,
    training_data=train_data_loader,
    test_data=test_data_loader,
    progress=True,
    inversion_method=InversionMethod.Cg,
    hessian_regularization=1e-1,
    maxiter=200,
)
```


### Shapley Values
The steps required to compute values for your samples are:

1. Create a `Dataset` object with your train and test splits.
2. Create an instance of a `SupervisedModel` (basically any sklearn compatible
   predictor)
3. Create a `Utility` object to wrap the Dataset, the model and a scoring
   function.
4. Use one of the methods defined in the library to compute the values.

This is how it looks for *Truncated Montecarlo Shapley*, an efficient method for
Data Shapley values:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from pydvl.value import *

data = Dataset.from_sklearn(load_breast_cancer(), train_size=0.7)
model = LogisticRegression()
u = Utility(model, data, Scorer("accuracy", default=0.0))
values = compute_shapley_values(
    u,
    mode=ShapleyMode.TruncatedMontecarlo,
    done=MaxUpdates(100) | AbsoluteStandardError(threshold=0.01),
    truncation=RelativeTruncation(u, rtol=0.01),
)
```

For more instructions and information refer to [Getting
Started](https://appliedAI-Initiative.github.io/pyDVL/10-getting-started.html) in
the documentation. We provide several
[examples](https://appliedAI-Initiative.github.io/pyDVL/examples/index.html)
with details on the algorithms and their applications.

## Caching

pyDVL offers the possibility to cache certain results and
speed up computation. It uses [Memcached](https://memcached.org/) For that.

You can run it either locally or, using
[Docker](https://www.docker.com/):

```shell
docker container run --rm -p 11211:11211 --name pydvl-cache -d memcached:latest
```

You can read more in the
[documentation](https://appliedAI-Initiative.github.io/pyDVL/getting-started/first-steps/#caching).

# Contributing

Please open new issues for bugs, feature requests and extensions. You can read
about the structure of the project, the toolchain and workflow in the [guide for
contributions](CONTRIBUTING.md).

# License

pyDVL is distributed under
[LGPL-3.0](https://www.gnu.org/licenses/lgpl-3.0.html). A complete version can
be found in two files: [here](LICENSE) and [here](COPYING.LESSER).

All contributions will be distributed under this license.

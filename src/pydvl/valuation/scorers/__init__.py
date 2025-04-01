"""
Scorers are a fundamental building block of many data valuation methods. They
are typically used by [ModelUtility][pydvl.valuation.utility.modelutility.ModelUtility]
and its subclasses to evaluate the quality of a model when trained on subsets of the
training data.

Scorers evaluate trained models in user-defined ways, and provide additional
information about themselves, like their range and default value, which can be used by
some data valuation methods (e.g. [Group Testing
Shapley][pydvl.valuation.methods.gt_shapley]) to estimate the number of samples required
for a certain quality of approximation.

!!! example "Named scorer"
    It is possible to use all named scorers from scikit-learn.

    ```python
    from pydvl.valuation import Dataset, SupervisedScorer

    train, test = Dataset.from_arrays(X, y, train_size=0.7)
    model = SomeSKLearnModel()
    scorer = SupervisedScorer("accuracy", test, default=0, range=(0, 1))
    ```

!!! example "Model scorer"
    It is also possible to use the `score()` function from the model if it defines one:

    ```python
    from pydvl.valuation import Dataset, SupervisedScorer

    train, test = Dataset.from_arrays(X, y, train_size=0.7)
    model = SomeSKLearnModel()
    scorer = SupervisedScorer(model, test, default=0, range=(-np.inf, 1))
    ```

For more examples see all submodules.
"""

from .base import *
from .classwise import *
from .supervised import *
from .utils import *

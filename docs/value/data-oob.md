from valuation.test_interface import datasets---
title: Data-OOB
---

# Data valuation for bagged models with Data-OOB

Data-OOB [@kwon_dataoob_2023] is a method for valuing data used to train bagged
models. It defines value as the out-of-bag (OOB) performance estimate for the
model, overcoming the computational bottleneck of Shapley-based data valuation
methods: Instead of fitting a large number of models to accurately estimate
marginal contributions like Shapley-based methods, Data-OOB evaluates each weak
learner in an ensemble over samples it hasn't seen during training, and averages
the performance across all weak learners.

More precisely, for a bagging model with $B$ estimators $\hat{f}_b, b \in [B]$,
we define $w_{bj}$ as the number of times that the $j$-th sample is in the
training set of the $b$-th estimator. For a **fixed** choice of bootstrapped
training sets, the Data-OOB value of sample $(x_i, y_i)$ is defined as:
 
$$ \psi_i := \frac{\sum_{b=1}^{B}\mathbb{1}(w_{bi}=0)T(y_i, 
   \hat{f}_b(x_i))}{\sum_{b=1}^{B} \mathbb{1} (w_{bi}=0)},
$$

where $T: Y \times Y \rightarrow \mathbb{R}$ is a score function that represents
the goodness of weak learner $\hat{f}_b$ at the $i$-th datum $(x_i, y_i)$.

$\psi$ can therefore be interpreted as a per-sample partition of the standard
OOB error estimate for a bagging model, which is: $\frac{1}{n} \sum_{i=1}^n
\psi_i$.

## Computing values

The main class is
[DataOOBValuation][pydvl.valuation.methods.data_oob.DataOOBValuation]. It takes
a *fitted* bagged model and uses data precomputed during training to calculate
the values. It is therefore very fast, and can be used to value large datasets.

This is how you would use it with a [[RandomForestClassifier]]:

```python
from sklearn.ensemble import RandomForestClassifier
from pydvl.valuation import DataOOBValuation, Dataset

train, test = Dataset(...), Dataset(...)
model = RandomForestClassifier(...)
model.fit(train.x, train.y)
valuation = DataOOBValuation(model)
valuation.fit(train)
values = valuation.values()
```

`values` is then a [ValuationResult][pydvl.valuation.result.ValuationResult] to
be used for data inspection, cleaning, etc.

Data-OOB is not limited to sklearn's [[RandomForest]], but can be used with
any bagging model that defines the attribute `estimators_`  after fitting and
makes the list of bootstrapped samples available in some way. This includes
[[BaggingRegressor]], [[BaggingClassifier]], [[ExtraTreesClassifier]],
[[ExtraTreesRegressor]] and [[IsolationForest]].

## Bagging arbitrary models

Through [[BaggingClassifier]] and [[BaggingRegressor]], one can compute values
for any model that can be bagged. Bagging in itself is not necessarily always
beneficial, and there are cases where it can be detrimental. However, for data
valuation we are not interested in the performance of the bagged model, but in
the valuation coming out of it, which can then be used to work on the original
model and data.

```python
from sklearn.ensemble import BaggingClassifier
from pydvl.valuation import DataOOBValuation, Dataset

train, test = Dataset(...), Dataset(...)
model = BaggingClassifier(
    estimator=KNeighborsClassifier(n_neighbors=10),
    n_estimators=20)
model.fit(train.x, train.y)
valuation = DataOOBValuation(model)
valuation.fit(train)
values = valuation.values()
values.sort()
low_values = values[:int(0.05*len(train))]  # select lowest 5%

# Inspect the data with lowest values:
...
```

### Off-topic: When not to use bagging as the main model

Here are some guidelines for when bagging might not be beneficial:

1. **Low-Variance Models**: Models like linear regression, support vector
   machines, or other inherently stable algorithms typically have low variance.
   In such cases, bagging may not provide significant benefits and could even
   increase computational cost unnecessarily.

2. **When the Model Is Already Highly Regularized**: If a model is regularized
   (e.g., Lasso, Ridge, or Elastic Net), it is already tuned to avoid
   overfitting and reduce variance. Bagging might be redundant or offer marginal
   improvements at best.

3. **When Data Is Limited**: Bagging works by creating multiple subsets of the
   data via bootstrapping. If the dataset is too small, the bootstrap samples
   might overlap significantly or exclude important patterns, reducing the
   effectiveness of the approach.

4. **When Features Are Highly Correlated**: If features are highly correlated,
   the individual models trained on different bootstrap samples may end up being
   too similar. This limits the diversity among the models, reducing the
   effectiveness of bagging.
   
5. **For Models That Are Not Easily Overfitted**: If the base model doesn't
   overfit the data (e.g., nearest neighbors or models with strong pruning
   mechanisms like heavily regularized decision trees), bagging might not yield
   substantial variance reduction.

6. **When Computational Resources Are Limited**: Bagging increases computational
   cost because it involves training multiple models. For computationally
   expensive algorithms or resource-constrained scenarios, bagging may be
   impractical.

7. **When Interpretability Is Critical**: Bagging produces an ensemble of
   models, which makes the overall model less interpretable compared to a single
   model. If interpretability is crucial for the application, bagging might not
   be suitable.

8. **When the Model Does Not Benefit from Resampling**: Some models, such as
   nearest neighbors, depend heavily on the full dataset. Subsampling through
   bootstrapping can reduce their performance instead of improving it.

9. **When Overfitting Is Already Controlled by Other Means**: If
   cross-validation, regularization, or pruning is already effectively
   controlling overfitting, bagging may offer little additional benefit.

## Transferring values

As with any other valuation method, you can transfer the values to a different
model, and given the efficiency of Data-OOB, this can be done very quickly. A
simple workflow is to compute values using a random forest, then use them to
inspect the data and clean it, and finally train a more complex model on the
cleaned data. Whether this is a valid idea or not will depend on the specific
dataset.

```python
...
```
...

## A comment about sampling

One might fear that there is a problem because the computation of the value
$\psi_i$ requires at least some bootstrap samples *not* to include the $i$-th
sample. But we can see that this is rarely an issue, and its probability of
happening can be easily computed: For a training set of size $n$ and
bootstrapping sample size $m \le n$, the probability that index $i$ is not
included in a bootstrap sample is $\prod_{j=1}^m \mathbb{P}(i \text{ is not
drawn at pos. } j) = (1 - 1/n)^m$, i.e. for each of the $m$ draws, the number is
not picked (for $m=n$ this converges to $1/e \approx 0.368$). The probability
that across $B$ bootstrapped samples a point is not included is therefore $(1  -
1/n)^{mB}$, which is typically extremely low.

Incidentally, this allows us to estimate the estimated number of unique indices
in a bootstrap sample of size $m$ as $m(1 - 1/n)^m$.


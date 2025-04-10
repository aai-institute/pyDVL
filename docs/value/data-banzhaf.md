---
title: Data Banzhaf
alias:
  name: data-banzhaf-intro
  title: Data Banzhaf
---

# Data Banzhaf semi-values  { #data-banzhaf-intro }

As noted in the section [Problems of Data Values][problems-of-data-values], the
Shapley value can be very sensitive to variance in the utility function. For
machine learning applications, where the utility is typically the performance
when trained on a set $S \subset D$, this variance is often largest for smaller
subsets $S$. It is therefore reasonable to try reducing the relative
contribution of these subsets with adequate weights.

One such choice of weights is the Banzhaf index, which assigns a constant weight:
 
$$
w(k) := 2^{-(n-1)},
$$

for all set sizes $k$. The intuition for picking a constant weight is that for
any choice of weight function $w$, one can always construct a utility with
higher variance where $w$ is greater. Therefore, in a worst-case sense, the best
one can do is to pick a constant weight.

The authors of [@wang_data_2023] show that Banzhaf indices are more robust to
variance in the utility function than Shapley and Beta Shapley values. They are
available in pyDVL through
[BanzhafValuation][pydvl.valuation.methods.banzhaf.BanzhafValuation]:

```python
from joblib import parallel_config
from pydvl.valuation import (
    ModelUtility, Dataset, SupervisedScorer, PermutationSampler
)
from pydvl.valuation.methods.banzhaf import BanzhafValuation
from pydvl.valuation.stopping import MinUpdates

train, test = Dataset.from_arrays(...)
model = ...
scorer =  SupervisedScorer(model, test, default=0.0)
utility = ModelUtility(model, scorer)
sampler = PermutationSampler()
valuation = BanzhafValuation(utility, sampler, MinUpdates(1000))
with parallel_config(n_jobs=16):
    valuation.fit(train)
```

## Data Banzhaf with MSR sampling  {  #msr-banzhaf-intro }

Wang et al. propose a more sample-efficient method for computing Banzhaf 
semi-values in their paper *Data Banzhaf: A Robust Data Valuation Framework 
for Machine Learning* [@wang_data_2023]. This method updates all semi-values
per each evaluation of the utility (i.e. per model training) based on whether a 
specific data point was included in the data subset or not. The expression 
for computing the semi-values is

$$
\hat{\phi}_{MSR}(i) = \frac{1}{|\mathbf{S}_{\ni i}|} \sum_{S \in 
\mathbf{S}_{\ni i}} U(S) - \frac{1}{|\mathbf{S}_{\not{\ni} i}|} 
\sum_{S \in \mathbf{S}_{\not{\ni} i}} U(S)
$$

where $\mathbf{S}_{\ni i}$ are the subsets that contain the index $i$ and 
$\mathbf{S}_{\not{\ni} i}$ are the subsets not containing the index $i$.

pyDVL provides a sampler for this method, called
[MSRSampler][pydvl.valuation.samplers.msr.MSRSampler], which can be combined
with any valuation method, including
[BanzhafValuation][pydvl.valuation.methods.banzhaf.BanzhafValuation]. However,
because the sampling probabilities induced by MSR coincide with Banzhaf indices,
it is preferred to use the dedicated class
[MSRBanzhafValuation][pydvl.valuation.methods.banzhaf.MSRBanzhafValuation]. For
more on this subject see [[semi-values-sampling]].

??? Example "MSR Banzhaf values"

    ```python
    from joblib import parallel_config
    from pydvl.valuation import ModelUtility, Dataset, SupervisedScorer
    from pydvl.valuation.methods.banzhaf import MSRBanzhafValuation
    from pydvl.valuation.stopping import MaxSamples
    
    train, test = Dataset.from_arrays(...)
    model = ...
    scorer = SupervisedScorer(model, test, default=0.0)
    utility = ModelUtility(model, scorer)
    valuation = MSRBanzhafValuation(utility, MaxSamples(1000), batch_size=64)
    with parallel_config(n_jobs=16):
        valuation.fit(train)
    ```
    Note how we pass batch size directly to the valuation method, which does
    not take a sampler since it uses MSR sampling internally.


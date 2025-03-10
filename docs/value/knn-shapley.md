---
title: KNN-Shapley
alias:
  name: knn-shapley-intro
  text: Introduction to KNN-Shapley
---

# KNN-Shapley { #knn-shapley-intro }

Using the generic class
[ShapleyValuation][pydvl.valuation.methods.shapley.ShapleyValuation] together
with a
[PermutationSampler][pydvl.valuation.samplers.permutation.PermutationSampler]
to compute Shapley values for a KNN model has a time complexity of 
$O(n^2 \log^2 n).$[^1]

However, it is possible to exploit the local structure of K-Nearest Neighbours
to drastically reduce the amount of utility evaluations and bring complexity
down to $O(n \log n)$ per test point. The key insight is that  because no sample
besides the $K$ closest affects the score, most can be ignored, allowing for a
recursive formula of the complexity mentioned. This method was introduced by
[@jia_efficient_2019a] in two forms, an exact (available with
[KNNShapleyValuation][pydvl.valuation.methods.knn_shapley.KNNShapleyValuation])
and an approximate one.


## The exact computation

Define the utility by the likelihood of the right label:

$$ U (S, x_{\text{test}}) = \frac{1}{K}  \sum_{k = 1}^{\min K, |S|}
   \mathbb{I}[y_{\alpha_{k} (S)} = y_{\text{test}}] $$

where $\alpha_{k} (S)$ is the $k$-th closest point $x_{i} \in S$ to 
$x_{\text{test}}$. The SV of each $x_{i} \in 
D_{\text{train}}$ is computed exactly with the following recursion:

$$ s_{\alpha_{N}} = \frac{\mathbb{I}[y_{\alpha_{N}} =
   y_{\text{test}}]}{N}, $$

$$ s_{\alpha_{i}} = s_{\alpha_{i + 1}} + \frac{\mathbb{I}[y_{\alpha
   _{i}} = y_{\text{test}}] -\mathbb{I}[y_{\alpha_{i + 1}} =
   y_{\text{test}}]}{K}  \frac{\min \lbrace K, i \rbrace}{i}.  $$

The utilities are then averaged over all $x_{\text{test}} \in 
D_{\text{test}}$ to compute the total utility $U (S).$

pyDVL implements the exact method in
[KNNShapleyValuation][pydvl.valuation.methods.knn_shapley.KNNShapleyValuation].
Internally it iterates through each training point with a
[LOOSampler][pydvl.valuation.samplers.powerset.LOOSampler] and computes the
recursive formula.[^2]

??? Example "Computing exact KNN-Shapley values"
    ```python
    from joblib import parallel_config
    from pydvl.valuation import Dataset, KNNShapleyValuation
    from sklearn.neighbors import KNeighborsClassifier
    
    model = KNeighborsClassifier(n_neighbors=5)
    train, test = Dataset.from_arrays(...)
    valuation = KNNShapleyValuation(model, test, progress=True)
    
    with parallel_config(n_jobs=16):
        valuation.fit(train)
    ```

## Approximate computation

By using approximate KNN-based methods, it is possible to shave off a factor $n$
in the complexity of the exact method. As of v0.10.0 there is no implementation
in pyDVL for this technique.

[^1]: This is based on a Hoeffding bound and the time complexity of sorting $n$
      points, which is the asymptotic complexity of one utility evaluation of
      KNN [@jia_efficient_2019a].
[^2]: As already noted, it is possible to use the standard infrastructure instead
      although this is only useful for testing purposes To this end, we provide
      an implementation of the formula above in
      [KNNClassifierUtility][pydvl.valuation.utility.knn.KNNClassifierUtility].

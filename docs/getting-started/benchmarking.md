---
title: Benchmarking tasks
alias:
  name: benchmarking-tasks
---

# Benchmarking tasks  { #benchmarking-tasks }

Because the magnitudes of values or influences from different algorithms, or
datasets, are not comparable to each other, evaluation of the methods is
typically done with _downstream tasks_.

## Benchmarking valuation methods  { #benchmarking-valuation-methods }

Data valuation is particularly useful for data selection, pruning and
inspection in general. For this reason, the most common benchmarks are
**data removal** and **noisy label detection**.

### High-value point removal

After computing the values for all data in $T = \{ \mathbf{z}_i : i = 1,
\ldots, n \}$, the set is sorted by decreasing value. We denote by $T_{[i :]}$
the sorted sequence of points $(\mathbf{z}_i, \mathbf{z}_{i + 1},
\ldots, \mathbf{z}_n)$ for $1 \leqslant i \leqslant n$. Now train
successively $f_{T [i :]}$ and compute its accuracy $a_{T_{[i :]}}
(D_{\operatorname{test}})$ on the held-out test set, then plot all numbers. By
using $D_{\operatorname{test}}$ one approximates the expected accuracy drop on
unseen data. Because the points removed have a high value, one expects
performance to drop visibly wrt. a random baseline.

### Low-value point removal

The complementary experiment removes data in increasing order, with the lowest
valued points first. Here one expects performance to increase relatively to
randomly removing points before training. Additionally, every real dataset will
include slightly out-of-distribution points, so one should also expect an
absolute increase in performance when some of the lowest valued points are
removed.

### Value transfer

This experiment explores the extent to which data values computed with one
(cheap) model can be transferred to another (potentially more complex) one.
Different classifiers are used as a source to calculate data values. These
values are then used in the point removal tasks described above, but using a
different (target) model for evaluation of the accuracies $a_{T [i :]}$. A
multi-layer perceptron is added for evaluation as well.

### Noisy label detection

This experiment tests the ability of a method to detect mislabeled instances in
the data. A fixed fraction $\alpha$ of the training data are picked at random
and their labels flipped. Data values are computed, then the $\alpha$-fraction
of lowest-valued points are selected, and the overlap with the subset of flipped
points is computed. This synthetic experiment is however hard to put into
practical use, since the fraction $\alpha$ is of course unknown in practice.

### Rank stability

Introduced in [@wang_data_2023], one can  look at how stable the top $k$% of
the values is across runs. Rank stability of a method is necessary but not
sufficient for good results. Ideally one wants to identify high-value points
reliably (good precision and recall) and consistently (good rank stability).

## Benchmarking Influence function methods  { #benchmarking-influence-methods }

!!! Todo
    This section is basically a stub

Although in principle one can compute the average influence over the test set
and run the same tasks as above, because influences are computed for each pair
of training and test sample, they typically require different experiments to
compare their efficacy.

### Approximation quality

The biggest difficulty when computing influences is the approximation of the
inverse Hessian-vector product. For this reason one often sees in the literature
the quality of the approximation to LOO as an indicator of performance, the
exact Influence Function being a first order approximation to it. However, as
shown by [@bae_if_2022], the different approximation errors ensuing for lack of
convexity, approximate Hessian-vector products and so on, lead to this being a
poor benchmark overall.

### Data re-labelling

[@kong_resolving_2022] introduce a method using IFs to re-label harmful training
samples in order to improve accuracy. One can then take the obtained improvement
as a measure of the quality of the IF method.

### Post-hoc fairness adjustment

Introduced in [@...], the idea is to compute influences over a carefully
selected fair set, and using them to re-weight the training data.





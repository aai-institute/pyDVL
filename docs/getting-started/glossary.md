---
title: Glossary
alias:
  name: glossary
  text: Glossary
search:
  boost: 10
---

# Glossary

This glossary is meant to provide only brief explanations of each term, helping
to clarify the concepts and techniques used in the library. For more detailed
information, please refer to the relevant literature or resources.

!!! warning
    This glossary is still a work in progress. Pull requests are welcome!

## Data valuation terms

### Beta-Shapley { #glossary-beta-shapley }

Beta-Shapley is a semi-value method that defines weights using a beta
distribution, thus effectively implementing an importance sampling scheme that
puts weight on marginal utilities for sets of certain sizes, as a function of
the parameters of the beta distribution. Introduced in [@kwon_beta_2022].

  * [Implementation][pydvl.valuation.methods.beta_shapley.BetaShapleyValuation]
  * [Documentation][beta-shapley-intro]

### Class-wise Shapley { #glossary-class-wise-shapley }

Class-wise Shapley is a Shapley valuation method which introduces a utility
function that balances in-class, and out-of-class accuracy, with the goal of
favoring points that improve the model's performance on the class they belong
to. It is estimated to be particularly useful in imbalanced datasets, but more
research is needed to confirm this.
Introduced by [@schoch_csshapley_2022].

 * [Implementation
   ][pydvl.valuation.methods.classwise_shapley.ClasswiseShapleyValuation]
 * [Documentation][classwise-shapley-intro]


### Data-Banzhaf { #glossary-data-banzhaf }

Data-Banzhaf is a semi-value method that uses the Banzhaf value to determine the
contribution of each data point to the model's performance. Its constant weights
are a somwewhat effective importance sampling scheme that reduces the variance of
the marginal utility estimates, especially for small subsets. It is most
efficient when used in conjunction with the [MSR][glossary-msr] sampler.
Introduced by [@wang_data_2023].

* [Implementation
  ][pydvl.valuation.methods.banzhaf.BanzhafValuation]
* [Documentation][data-banzhaf-intro]


### Data-OOB { #glossary-data-oob }

Data-OOB is a method for valuing data points for a bagged model using its
out-of-bag performance estimate. It overcomes the computational bottleneck of
Shapley-based methods by evaluating each weak learner in an ensemble over
samples it hasn't seen during training, and averaging the performance across
all weak learners.
Introduced in [@kwon_dataoob_2023].

 * [Implementation][pydvl.valuation.methods.data_oob.DataOOBValuation]
 * [Documentation][data-oob-intro]


### Data Utility Learning { #glossary-data-utility-learning }

Data Utility Learning is a method that uses an ML model to learn the utility
function. Essentially, it learns to predict the performance of a model when
trained on a given set of indices from the dataset. The cost of training this
model is quickly amortized by avoiding costly re-evaluations of the original
utility.
Introduced by [@wang_improving_2022].

 * [Implementation][pydvl.valuation.utility.learning.DataUtilityLearning]
 * [Documentation][data-utility-learning-intro]

### Delta-Shapley { #glossary-delta-shapley }

Delta-Shapley is an approximation to Shapley value which uses a [stratified
sampling][glossary-stratified-sampling] distribution that picks set sizes based
on stability bounds for the machine learning model for which values are 
estimated. An additional clipping constraint saves computation by skipping
subset sizes (justified because of diminishing returns for model performance).
This introduces a difference to Shapley value by a multiplicative factor that
should not affect ranking. Introduced in [@watson_accelerated_2023].

 * [Implementation][pydvl.valuation.methods.delta_shapley.DeltaShapleyValuation]
 * [Documentation][delta-shapley-intro]

### Game-theoretic Methods { #glossary-game-theoretic-methods }

Game-theoretic methods for data valuation are a class of techniques that
leverage concepts from cooperative game theory to assign values to data points
based on their contributions to the overall performance of an ML model. Salient
examples are [Shapley value][glossary-shapley-value], [Data
Banzhaf][glossary-data-banzhaf], and [Least Core][glossary-least-core].


### Group Testing { #glossary-group-testing }

Group Testing is a strategy for identifying characteristics within groups of
items efficiently, by testing groups rather than individuals to quickly narrow
down the search for items with specific properties. The idea was introduced into
data valuation by [@jia_efficient_2019a], transforming the problem of computing
Shapley values into one of constraint satisfiability, with constraints given by
samples of the utility, with a carefully designed sampling distribution.

 * [Implementation][pydvl.valuation.methods.gt_shapley.GroupTestingShapleyValuation]
 * [Documentation][group-testing-shapley-intro]

### KNN-Shapley { #glossary-knn-shapley }

KNN-Shapley is a Shapley value method tailored to KNN models. By exploiting the
locality of KNN, it can reduce computation drastically wrt. the standard
Shapley value with Monte Carlo approximation. Introduced in
[@jia_efficient_2019a].

* [Implementation][pydvl.valuation.methods.knn_shapley.KNNShapleyValuation]
* [Documentation][knn-shapley-intro]


### Importance Sampling { #glossary-importance-sampling }

Importance Sampling is a technique used to estimate properties of a particular
distribution while only having samples generated from a different distribution.
This is achieved by re-weighting the samples according to their "sampled
importance", i.e. effectively dividing by the probability of sampling them.

Much of the work in model-based data valuation consists of finding a good
importance sampling distribution, and the weights to use for the sampled
subsets. The most common choices are uniform sampling, Beta distributions, and
Banzhaf indices.


### Least Core { #glossary-least-core }

The Least Core is a solution concept in cooperative game theory, referring to
the smallest set of payoffs to players that cannot be improved upon by any
coalition, ensuring stability in the allocation of value. In data valuation,
it implies solving a linear and a quadratic system whose constraints are
determined by the evaluations of the utility function on every subset of the
training data.
Introduced as data valuation method by [@yan_if_2021].

 * [Implementation][pydvl.valuation.methods.least_core.LeastCoreValuation]
 * [Documentation][least-core-values]


### Leave-One-Out { #glossary-loo }

LOO in the context of data valuation refers to the process of evaluating the
impact of removing individual data points on the model's performance. The
value of a training point is defined as the marginal change in the model's
performance when that point is removed from the training set.

 * [Implementation][pydvl.valuation.methods.loo.LOOValuation]
 * [Documentation][loo-valuation-intro]


### Marginal utility { #glossary-marginal-utility }

In data valuation for ML, _marginal utility_ refers to the change in performance
of an ML model when a single data point is added to or removed from the training
set. In our documentation it is often denoted $\Delta_i(S) := U(S_{+i}) - U(S),$
where $S$ is a subset of the training set, $i$ is the index of the data point
to be added, and $U$ is the [utility function][glossary-utility-function].

* [Introduction to data valuation][data-valuation-intro]

### Marginal-based methods

Marginal-based methods are a class of data valuation techniques that define
value in terms of weighted averages of the [marginal 
utility][glossary-marginal-utility].

* [Introduction to data valuation][data-valuation-intro]


### Maximum Sample Reuse  { #glossary-msr }

MSR is a sampling method for data valuation that updates the value of every data
point in one sample. This method can achieve much faster convergence for
[Data Banzhaf][glossary-data-banzhaf] since the sampling distribution
"coincides" with the Banzhaf coefficients. In principle, it can be used
with any [semi-value][pydvl.valuation.methods.semivalue.SemivalueValuation] by
setting the sampler to [MSRSampler][pydvl.valuation.samplers.msr.MSRSampler],
but it's most effective when used with the Banzhaf semi-value via
[MSRBanzhafValuation][pydvl.valuation.methods.banzhaf.MSRBanzhafValuation].
Introduced by [@wang_data_2023]

* [Sampler][pydvl.valuation.samplers.msr.MSRSampler]
* [MSRBanzhafValuation][pydvl.valuation.methods.banzhaf.MSRBanzhafValuation]


### Monte Carlo Least Core  { #glossary-mclc }

MCLC is a variation of the Least Core that uses a reduced amount of
constraints, sampled randomly from the powerset of the training data.
Introduced by [@yan_if_2021].

 * [Implementation][pydvl.valuation.methods.least_core.MonteCarloLeastCoreValuation]
 * [Documentation][monte-carlo-least-core]

### Monte Carlo Shapley  { #glossary-monte-carlo-shapley }

MCS estimates the Shapley Value using a Monte Carlo approximation to the sum
over subsets of the training set. This reduces computation to polynomial time
at the cost of accuracy, but this loss is typically irrelevant for downstream
applications in ML.
Introduced into data valuation by [@ghorbani_data_2019].

 * [Implementation][pydvl.valuation.methods.shapley.ShapleyValuation]
 * [Documentation][monte-carlo-combinatorial-shapley-intro]


### Point-removal experiment  { #glossary-point-removal-experiment }

A point-removal experiment is a benchmarking task in data valuation where the
quality of a valuation method is measured through the impact of incrementally
removing data points on the model's performance. The points are removed in
order of their value, and the performance is evaluated on a fixed validation set.

 * [Benchmarking valuation methods][benchmarking-valuation-methods].


### Rank correlation  { #glossary-rank-correlation }

Rank correlation is a simple way of measuring the stability of estimates for the
values of a training set. It is computed as the Spearman correlation between the
values of two different runs of the same method, after changing some
hyperparameter like random seed, number of updates during a single run, etc.

 * [Benchmarking valuation methods][benchmarking-valuation-methods].


### Shapley Value  { #glossary-shapley-value }

Shapley Value is a concept from cooperative game theory that allocates payouts
to players based on their contribution to the total payoff. In data valuation,
players are data points. The method assigns a value to each data point based
on a weighted average of its marginal contributions to the model's performance 
when trained on each subset of the training set. This requires
$\mathcal{O}(2^{n-1})$ re-trainings of the model, which is infeasible for even
trivial data set sizes, so one resorts to approximations like TMCS.
Introduced into data valuation by [@ghorbani_data_2019].

 * [Implementation][pydvl.valuation.methods.shapley]
 * [Documentation][shapley-valuation-intro]


### Stratified sampling { #glossary-stratified-sampling }

In pyDVL, a [stratified sampler][pydvl.valuation.samplers.stratified] is one
that first chooses a subset size $k$ following some distribution
$\mathcal{L}_k$ over $\{0,...,n-1\},$ then samples a subset of that size
uniformly at random from the powerset of ${N_{-i}}:$

1. Sample $k \sim \mathcal{L}_k,$
2. Sample $S \sim \mathcal{U}(2^{N_{-i}}).$

If we denote by $\mathcal{L}$ the law for this two stage procedure, then one has
that the [Shapley value][glossary-shapley-value] is the expectation over this
distribution:

$$v_\text{sh}(i) = \mathbb{E}_{S \sim \mathcal{L}}[\Delta_i(S)].$$

One can try to reduce variance or obtain different semi-values by choosing
$\mathcal{L}_k$ differently, or combining it with any semivalue. See the links
below.

 * [Data Shapley with a uniform stratified sampler][stratified-shapley-value]
 * [Sampler implementation][pydvl.valuation.samplers.stratified]
 * [Variance-Reduced Data Shapley][glossary-vrds]
 * [$\delta$-Shapley][glossary-delta-shapley] 


### Truncated Monte Carlo Shapley  { #glossary-tmcs }

TMCS is an efficient approach to estimating the Shapley Value using a
truncated version of the Monte Carlo method with permutation sampling.
Introduced by [@ghorbani_data_2019].

 * [Implementation][pydvl.valuation.methods.shapley.ShapleyValuation]
 * [Documentation][permutation-shapley-intro]


### Utility function { #glossary-utility-function }

The _utility function_ in ML data valuation is a measure of the performance of a
model trained on a subset of the training data. This can be a metric like
accuracy, F1 score, or any other performance measure. It is typically measured
wrt. to a fixed [valuation set][glossary-valuation-set] (sometimes we use the
terms _test set_ or _validation set_ interchangeably when no confusion is
possible, but it should be a different, held-out set.

In our documentation the utility is denoted $U$, and is assumed to be a function
defined over sets (hence invariant wrt. permutation of data indices):
$U:2^{N} \to \mathbb{R}$, where $N$ is the index set of the training data, which
we identify with the data themselves.

* [Introduction to data valuation][data-valuation-intro]

### Valuation set { #glossary-valuation-set }

The _valuation set_ is a held-out subset of data used to evaluate the utility of
a model trained on the training set. Sometimes, when there is no confusion, we
use the terms _test set_ or _validation set_ interchangeably, but it should be
a different, held-out set.

Note that computing a score (loss) over a fixed set is typically a poor
approximation to the true score of the model, i.e. to its expected score on
unseen data. This problem might be alleviated with some form of cross-validation,
but we haven't explored this possibility in pyDVL.

### Variance-Reduced Data-Shapley { #glossary-vrds }

A [stratified sampling][glossary-stratified-sampling]-based approach to estimate
Shapley values that uses a simple deterministic heuristic for sample sizes,
which in particular does not depend on run-time variance estimates. A good
default choice is based on the harmonic function. Introduced in
[@wu_variance_2023].

* [Implementation][pydvl.valuation.samplers.stratified.VRDSSampler]


### Weighted Accuracy Drop  { #glossary-wad }

WAD is a metric to evaluate the impact of sequentially removing data points on
the performance of a machine learning model, weighted by their rank, i.e. by the
time at which they were removed.
Introduced by [@schoch_csshapley_2022].

---

## Influence function terms

### Arnoldi Method { #glossary-arnoldi }

The Arnoldi method approximately computes eigenvalue, eigenvector pairs of
a symmetric matrix. For influence functions, it is used to approximate
the [iHVP][glossary-iHVP].
Introduced by [@schioppa_scaling_2022] in the context of influence functions.

  * [Implementation (torch)
    ][pydvl.influence.torch.influence_function_model.ArnoldiInfluence]
  * [Documentation (torch)][arnoldi]


### Block Conjugate Gradient { #glossary-block-cg }

A blocked version of [CG][glossary-cg], which solves several linear
systems simultaneously. For Influence Functions, it is used to
approximate the [iHVP][glossary-iHVP].

 * [Implementation (torch)
   ][pydvl.influence.torch.influence_function_model.CgInfluence]
 * [Documentation (torch)][cg]

### Conjugate Gradient { #glossary-cg }

CG is an algorithm for solving linear systems with a symmetric and
positive-definite coefficient matrix. For Influence Functions, it is used to
approximate the [iHVP][glossary-iHVP].

 * [Implementation
   (torch)][pydvl.influence.torch.influence_function_model.CgInfluence]
 * [Documentation (torch)][cg]


### Eigenvalue-corrected Kronecker-Factored Approximate Curvature

EKFAC builds on [K-FAC][glossary-k-fac] by correcting for the approximation
errors in the eigenvalues of the blocks of the Kronecker-factored approximate
curvature matrix. This correction aims to refine the accuracy of natural
gradient approximations, thus potentially offering better training efficiency
and stability in neural networks.

 * [Implementation
   (torch)][pydvl.influence.torch.influence_function_model.EkfacInfluence]
 * [Documentation (torch)][eigenvalue-corrected-k-fac]



### Influence Function { #glossary-influence-function }

The Influence Function measures the impact of a single data point on a
statistical estimator. In machine learning, it's used to understand how much a
particular data point affects the model's prediction.
Introduced into data valuation by [@koh_understanding_2017].

 * [Documentation][influence-function]

### Inverse Hessian-vector product { #glossary-iHVP }

iHVP is the operation of calculating the product of the inverse Hessian matrix
of a function and a vector, without explicitly constructing nor inverting the
full Hessian matrix first. This is essential for influence function computation.

### Kronecker-Factored Approximate Curvature { #glossary-k-fac }

K-FAC is an optimization technique that approximates the Fisher Information
matrix's inverse efficiently. It uses the Kronecker product to factor the
matrix, significantly speeding up the computation of natural gradient updates
and potentially improving training efficiency.


### Linear-time Stochastic Second-order Algorithm { #glossary-lissa }

LiSSA is an efficient algorithm for approximating the inverse Hessian-vector
product, enabling faster computations in large-scale machine learning
problems, particularly for second-order optimization.
For Influence Functions, it is used to
approximate the [iHVP][glossary-iHVP].
Introduced by [@agarwal_secondorder_2017].

 * [Implementation (torch)
   ][pydvl.influence.torch.influence_function_model.LissaInfluence]
 * [Documentation (torch)
   ][linear-time-stochastic-second-order-approximation-lissa]


### Nyström Low-Rank Approximation  { #glossary-nystroem }

The Nyström approximation computes a low-rank approximation to a symmetric
positive-definite matrix via random projections. For influence functions, 
it is used to approximate the [iHVP][glossary-iHVP].
Introduced as sketch and solve algorithm in [@hataya_nystrom_2023], and as
preconditioner for [PCG][glossary-preconditioned-cg] in
[@frangella_randomized_2023].

 * [Implementation Sketch-and-Solve
   (torch)][pydvl.influence.torch.influence_function_model.NystroemSketchInfluence]
 * [Documentation Sketch-and-Solve (torch)][nystrom-sketch-and-solve]
 * [Implementation Preconditioner
   (torch)][pydvl.influence.torch.preconditioner.NystroemPreconditioner]


### Preconditioned Block Conjugate Gradient  { #glossary-preconditioned-block-cg }

A blocked version of [PCG][glossary-preconditioned-cg], which solves 
several linear systems simultaneously. For Influence Functions, it is used to
approximate the [iHVP][glossary-iHVP].

 * [Implementation CG (torch)
   ][pydvl.influence.torch.influence_function_model.CgInfluence]
 * [Implementation Preconditioner
   (torch)][pydvl.influence.torch.preconditioner.Preconditioner]
 * [Documentation (torch)][cg]

### Preconditioned Conjugate Gradient  { #glossary-preconditioned-cg }

A preconditioned version of [CG][glossary-cg] for improved
convergence, depending on the characteristics of the matrix and the
preconditioner. For Influence Functions, it is used to
approximate the [iHVP][glossary-iHVP].

 * [Implementation CG (torch)
   ][pydvl.influence.torch.influence_function_model.CgInfluence]
 * [Implementation Preconditioner
   (torch)][pydvl.influence.torch.preconditioner.Preconditioner]
 * [Documentation (torch)][cg]


---

## Other terms

### Coefficient of Variation { #glossary-cv }

CV is a statistical measure of the dispersion of data points in a data series
around the mean, expressed as a percentage. It's used to compare the degree of
variation from one data series to another, even if the means are drastically
different.


### Constraint Satisfaction Problem { #glossary-csp }

A CSP involves finding values for variables within specified constraints or
conditions, commonly used in scheduling, planning, and design problems where
solutions must satisfy a set of restrictions.

### Out-of-Bag { #glossary-oob }

OOB refers to data samples in an ensemble learning context (like random forests)
that are not selected for training a specific model within the ensemble. These
OOB samples are used as a validation set to estimate the model's accuracy,
providing a convenient internal cross-validation mechanism.

### Machine Learning Reproducibility Challenge { #glossary-mlrc }

The [MLRC](https://reproml.org/) is an initiative that encourages the
verification and replication of machine learning research findings, promoting
transparency and reliability in the field. Papers are published in
[Transactions on Machine Learning Research](https://jmlr.org/tmlr/) (TMLR).


### Point removal task  { #glossary-point-removal-task }

A task in data valuation where the quality of a valuation method is measured
through the impact of incrementally removing data points on the model's
performance, where the points are removed in order of their value. See

 * [Implementation][pydvl.reporting.point_removal.run_removal_experiment]
 * [Benchmarking tasks][benchmarking-tasks]

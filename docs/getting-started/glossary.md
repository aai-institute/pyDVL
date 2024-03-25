# Glossary

This glossary is meant to provide only brief explanations of each term, helping
to clarify the concepts and techniques used in the library. For more detailed
information, please refer to the relevant literature or resources.

!!! warning
    This glossary is still a work in progress. Pull requests are welcome!

Terms in data valuation and influence functions:

### Arnoldi Method

The Arnoldi method approximately computes eigenvalue, eigenvector pairs of
a symmetric matrix. For influence functions, it is used to approximate
the [iHVP][inverse-hessian-vector-product].

Introduced by [@schioppa_scaling_2021] in the context of influence functions.
[Implementation (torch)
][pydvl.influence.torch.influence_function_model.ArnoldiInfluence].
[Documentation (torch)][arnoldi].

### Block Conjugate Gradient

A blocked version of [CG][conjugate-gradient], which solves several linear
systems simultaneously. For Influence Functions, it is used to
approximate the [iHVP][inverse-hessian-vector-product].
[Implementation (torch)][pydvl.influence.torch.influence_function_model.CgInfluence].
[Documentation (torch)][cg]

### Class-wise Shapley

Class-wise Shapley is a Shapley valuation method which introduces a utility
function that balances in-class, and out-of-class accuracy, with the goal of
favoring points that improve the model's performance on the class they belong
to. It is estimated to be particularly useful in imbalanced datasets, but more
research is needed to confirm this.
Introduced by [@schoch_csshapley_2022].
[Implementation][pydvl.value.shapley.classwise.compute_classwise_shapley_values].
[Documentation][class-wise-shapley].

### Conjugate Gradient

CG is an algorithm for solving linear systems with a symmetric and
positive-definite coefficient matrix. For Influence Functions, it is used to
approximate the [iHVP][inverse-hessian-vector-product].
[Implementation (torch)
][pydvl.influence.torch.influence_function_model.CgInfluence].
[Documentation (torch)][cg]

### Data Utility Learning

Data Utility Learning is a method that uses an ML model to learn the utility
function. Essentially, it learns to predict the performance of a model when
trained on a given set of indices from the dataset. The cost of training this
model is quickly amortized by avoiding costly re-evaluations of the original
utility.
Introduced by [@wang_improving_2022].
[Implementation][pydvl.utils.utility.DataUtilityLearning].

### Eigenvalue-corrected Kronecker-Factored Approximate Curvature

EKFAC builds on [K-FAC][kronecker-factored-approximate-curvature] by correcting
for the approximation errors in the eigenvalues of the blocks of the
Kronecker-factored approximate curvature matrix. This correction aims to refine
the accuracy of natural gradient approximations, thus potentially offering
better training efficiency and stability in neural networks.
[Implementation (torch)
][pydvl.influence.torch.influence_function_model.EkfacInfluence].
[Documentation (torch)][eigenvalue-corrected-k-fac].


### Group Testing

Group Testing is a strategy for identifying characteristics within groups of
items efficiently, by testing groups rather than individuals to quickly narrow
down the search for items with specific properties.
Introduced into data valuation by [@jia_efficient_2019a].
[Implementation][pydvl.value.shapley.gt.group_testing_shapley].

### Influence Function

The Influence Function measures the impact of a single data point on a
statistical estimator. In machine learning, it's used to understand how much a
particular data point affects the model's prediction.
Introduced into data valuation by [@koh_understanding_2017].
[[influence-function|Documentation]].

### Inverse Hessian-vector product

iHVP is the operation of calculating the product of the inverse Hessian matrix
of a function and a vector, without explicitly constructing nor inverting the
full Hessian matrix first. This is essential for influence function computation.

### Kronecker-Factored Approximate Curvature

K-FAC is an optimization technique that approximates the Fisher Information
matrix's inverse efficiently. It uses the Kronecker product to factor the
matrix, significantly speeding up the computation of natural gradient updates
and potentially improving training efficiency.

### Least Core

The Least Core is a solution concept in cooperative game theory, referring to
the smallest set of payoffs to players that cannot be improved upon by any
coalition, ensuring stability in the allocation of value. In data valuation,
it implies solving a linear and a quadratic system whose constraints are
determined by the evaluations of the utility function on every subset of the
training data.
Introduced as data valuation method by [@yan_if_2021].
[Implementation][pydvl.value.least_core.compute_least_core_values].

### Linear-time Stochastic Second-order Algorithm

LiSSA is an efficient algorithm for approximating the inverse Hessian-vector
product, enabling faster computations in large-scale machine learning
problems, particularly for second-order optimization.
For Influence Functions, it is used to
approximate the [iHVP][inverse-hessian-vector-product].
Introduced by [@agarwal_secondorder_2017].
[Implementation (torch)
][pydvl.influence.torch.influence_function_model.LissaInfluence].
[Documentation (torch)
][linear-time-stochastic-second-order-approximation-lissa].

### Leave-One-Out

LOO in the context of data valuation refers to the process of evaluating the
impact of removing individual data points on the model's performance. The
value of a training point is defined as the marginal change in the model's
performance when that point is removed from the training set.
[Implementation][pydvl.value.loo.loo.compute_loo].

### Monte Carlo Least Core

MCLC is a variation of the Least Core that uses a reduced amount of
constraints, sampled randomly from the powerset of the training data.
Introduced by [@yan_if_2021].
[Implementation][pydvl.value.least_core.compute_least_core_values].

### Monte Carlo Shapley

MCS estimates the Shapley Value using a Monte Carlo approximation to the sum
over subsets of the training set. This reduces computation to polynomial time
at the cost of accuracy, but this loss is typically irrelevant for downstream
applications in ML.
Introduced into data valuation by [@ghorbani_data_2019].
[Implementation][pydvl.value.shapley.montecarlo].
[[data-valuation|Documentation]].

### Nyström Low-Rank Approximation

The Nyström approximation computes a low-rank approximation to a symmetric
positive-definite matrix via random projections. For influence functions, 
it is used to approximate the [iHVP][inverse-hessian-vector-product].
Introduced as sketch and solve algorithm in [@hataya_nystrom_2023], and as
preconditioner for [PCG][preconditioned-conjugate-gradient] in
[@frangella_randomized_2023].
[Implementation Sketch-and-Solve (torch)
][pydvl.influence.torch.influence_function_model.NystroemSketchInfluence].
[Documentation Sketch-and-Solve (torch)][nystrom-sketch-and-solve].
[Implementation Preconditioner (torch)
][pydvl.influence.torch.pre_conditioner.NystroemPreConditioner].


### Point removal task

A task in data valuation where the quality of a valuation method is measured
through the impact of incrementally removing data points on the model's
performance, where the points are removed in order of their value. See
[Benchmarking tasks][benchmarking-tasks].

### Preconditioned Block Conjugate Gradient

A blocked version of [PCG][preconditioned-conjugate-gradient], which solves 
several linear systems simultaneously. For Influence Functions, it is used to
approximate the [iHVP][inverse-hessian-vector-product].
[Implementation CG (torch)
][pydvl.influence.torch.influence_function_model.CgInfluence]
[Implementation Preconditioner (torch)][pydvl.influence.torch.pre_conditioner]
[Documentation (torch)][cg]

### Preconditioned Conjugate Gradient

A preconditioned version of [CG][conjugate-gradient] for improved
convergence, depending on the characteristics of the matrix and the
preconditioner. For Influence Functions, it is used to
approximate the [iHVP][inverse-hessian-vector-product].
[Implementation CG (torch)
][pydvl.influence.torch.influence_function_model.CgInfluence]
[Implementation Preconditioner (torch)][pydvl.influence.torch.pre_conditioner]
[Documentation (torch)][cg]

### Shapley Value

Shapley Value is a concept from cooperative game theory that allocates payouts
to players based on their contribution to the total payoff. In data valuation,
players are data points. The method assigns a value to each data point based
on a weighted average of its marginal contributions to the model's performance 
when trained on each subset of the training set. This requires
$\mathcal{O}(2^{n-1})$ re-trainings of the model, which is infeasible for even
trivial data set sizes, so one resorts to approximations like TMCS.
Introduced into data valuation by [@ghorbani_data_2019].
[Implementation][pydvl.value.shapley.naive].
[[data-valuation|Documentation]].

### Truncated Monte Carlo Shapley

TMCS is an efficient approach to estimating the Shapley Value using a
truncated version of the Monte Carlo method, reducing computation time while
maintaining accuracy in large datasets.
Introduced by [@ghorbani_data_2019].
[Implementation][pydvl.value.shapley.montecarlo.permutation_montecarlo_shapley].
[[data-valuation|Documentation]].

### Weighted Accuracy Drop

WAD is a metric to evaluate the impact of sequentially removing data points on
the performance of a machine learning model, weighted by their rank, i.e. by the
time at which they were removed.
Introduced by [@schoch_csshapley_2022].

---

## Other terms

### Coefficient of Variation

CV is a statistical measure of the dispersion of data points in a data series
around the mean, expressed as a percentage. It's used to compare the degree of
variation from one data series to another, even if the means are drastically
different.


### Constraint Satisfaction Problem

A CSP involves finding values for variables within specified constraints or
conditions, commonly used in scheduling, planning, and design problems where
solutions must satisfy a set of restrictions.

### Out-of-Bag

OOB refers to data samples in an ensemble learning context (like random forests)
that are not selected for training a specific model within the ensemble. These
OOB samples are used as a validation set to estimate the model's accuracy,
providing a convenient internal cross-validation mechanism.

### Machine Learning Reproducibility Challenge

The [MLRC](https://reproml.org/) is an initiative that encourages the
verification and replication of machine learning research findings, promoting
transparency and reliability in the field. Papers are published in
[Transactions on Machine Learning Research](https://jmlr.org/tmlr/) (TMLR).

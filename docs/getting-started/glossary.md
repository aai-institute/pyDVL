# Glossary

This glossary is meant to provide only brief explanations of each term, helping
to clarify the concepts and techniques used in the library. For more detailed
information, please refer to the relevant literature or resources.

## Valuation

- **Class-wise Shapley:**
  Class-wise Shapley is a Shapley valuation method which introduces a utility
  function that balances in-class, and out-of-class accuracy, with the goal of
  favoring points that improve the model's performance on the class they belong
  to. It is estimated to be particularly useful in imbalanced datasets, but more
  research is needed to confirm this.
  Introduced by [@schoch_csshapley_2022].
  [Implementation][pydvl.value.shapley.classwise.compute_classwise_shapley_values].

- **Data Utility Learning:**
  Data Utility Learning is a method that uses an ML model to learn the utility
  function. Essentially, it learns to predict the performance of a model when
  trained on a given set of indices from the dataset. The cost of training this
  model is quickly amortized by avoiding costly re-evaluations of the original
  utility.
  Introduced by [@wang_improving_2022].
  [Implementation][pydvl.utils.utility.DataUtilityLearning].

- **Eigenvalue-corrected Kronecker-Factored Approximate Curvature**:
  EKFAC builds on K-FAC by correcting for the approximation errors in the
  eigenvalues of the blocks of the Kronecker-factored approximate curvature
  matrix. This correction aims to refine the accuracy of natural gradient
  approximations, thus potentially offering better training efficiency and
  stability in neural networks.
  [Implementation (torch)][pydvl.influence.torch.influence_function_model.EkfacInfluence].

- **Kronecker-Factored Approximate Curvature**:
  K-FAC is an optimization technique that approximates the Fisher Information
  matrix's inverse efficiently. It uses the Kronecker product to factor the
  matrix, significantly speeding up the computation of natural gradient updates
  and potentially improving training efficiency.

- **Group Testing:**
  Group Testing is a strategy for identifying characteristics within groups of
  items efficiently, by testing groups rather than individuals to quickly narrow
  down the search for items with specific properties.
  Introduced into data valuation by [@jia_efficient_2019a].
  [Implementation][pydvl.value.shapley.gt.group_testing_shapley].

- **Influence Function:**
  The Influence Function measures the impact of a single data point on a
  statistical estimator. In machine learning, it's used to understand how much a
  particular data point affects the model's prediction.
  Introduced into data valuation by [@koh_understanding_2017].
  [Documentation][influence-function].

- **inverse Hessian-vector product:**
  iHVP involves calculating the product of the inverse Hessian matrix of a
  function and a vector, which is essential in optimization and in computing
  influence functions efficiently.

- **Least Core:**
  The Least Core is a solution concept in cooperative game theory, referring to
  the smallest set of payoffs to players that cannot be improved upon by any
  coalition, ensuring stability in the allocation of value.
  Introduced as data valuation method by [@yan_if_2021].
  [Implementation][pydvl.value.least_core.common.lc_solve_problem].

- **Linear-time Stochastic Second-order Algorithm:**
  LiSSA is an efficient algorithm for approximating the inverse Hessian-vector
  product, enabling faster computations in large-scale machine learning
  problems, particularly for second-order optimization.
  Introduced by [@agarwal_secondorder_2017].
  [Implementation (torch)][pydvl.influence.torch.influence_function_model.LissaInfluence].

- **Leave-One-Out:**
  LOO in the context of data valuation refers to the process of evaluating the
  impact of removing individual data points on the model's performance. The
  value of a training point is defined as the marginal change in the model's
  performance when that point is removed from the training set.
  [Implementation][pydvl.value.loo.loo.compute_loo].

- **Monte Carlo Least Core:**
  MCLC is a variation of the Least Core that uses a reduced amount of
  constraints sampled randomly.
  Introduced by [@yan_if_2021].
  [Implementation][pydvl.value.least_core.compute_least_core_values].

- **Monte Carlo Shapley:**
  MCS estimates the Shapley Value using a Monte Carlo approximation to the sum
  over subsets of the training set. This reduces computation to polynomial time
  at the cost of accuracy, but this loss is typically irrelevant for downstream
  applications in ML.

- **Shapley Value:**
  Shapley Value is a concept from cooperative game theory that allocates payouts
  to players based on their contribution to the total payoff. In data valuation,
  players are data points. The method assigns a value to each data point based
  on a weighted average of its marginal contributions to the model's performance 
  when trained on each subset of the training set. This requires
  $\mathcal{O}(2^{n-1}$ evaluations of the model, which is infeasible for even
  trivial data set sizes, so one resorts to approximations like TMCS.

- **Truncated Monte Carlo Shapley:**
  TMCS is an efficient approach to estimating the Shapley Value using a
  truncated version of the Monte Carlo method, reducing computation time while
  maintaining accuracy in large datasets.
  Introduced by [@ghorbani_data_2019].
  [Implementation][pydvl.value.shapley.montecarlo.permutation_montecarlo_shapley].

- **Weighted Accuracy Drop:**
  WAD is a metric to evaluate the impact of sequentially removing data points on
  the performance of a machine learning model, weighted by their rank, i.e. by the
  time at which they were removed.
  Introduced by [@schoch_csshapley_2022].

## Other

- **Coefficient of Variation:**
  CV is a statistical measure of the dispersion of data points in a data series
  around the mean, expressed as a percentage. It's used to compare the degree of
  variation from one data series to another, even if the means are drastically
  different.

- **Conjugate Gradient:**
  CG is an algorithm for solving linear systems with a symmetric and
  positive-definite coefficient matrix. In machine learning, it's typically used
  for efficiently finding the minima of convex functions, when the direct
  computation of the Hessian is computationally expensive or impractical.

- **Constraint Satisfaction Problem:**
  A CSP involves finding values for variables within specified constraints or
  conditions, commonly used in scheduling, planning, and design problems where
  solutions must satisfy a set of restrictions.

- **Out-of-Bag:**
  OOB refers to data samples in an ensemble learning context (like random forests)
  that are not selected for training a specific model within the ensemble. These
  OOB samples are used as a validation set to estimate the model's accuracy,
  providing a convenient internal cross-validation mechanism.

- **Machine Learning Reproducibility Challenge:**
  The [MLRC](https://reproml.org/) is an initiative that encourages the
  verification and replication of machine learning research findings, promoting
  transparency and reliability in the field. Papers are published in
  [Transactions on Machine Learning Research](https://jmlr.org/tmlr/) (TMLR).

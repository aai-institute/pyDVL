---
title: Data Utility Learning
alias:
  name: data-utility-learning-intro
  title: Data Utility Learning
---

# Data Utility Learning  { #data-utility-learning-intro }

DUL [@wang_improving_2022] uses an ML model $\hat{u}$ to learn the utility function
$u:2^N \to \matbb{R}$ during the fitting phase of any valuation method. This
_utility model_ is trained with tuples $(S, U(S))$ for a certain warm-up period.
Then it is used instead of $u$ in the valuation method. The cost of training
$\hat{u}$ is quickly amortized by avoiding costly re-evaluations of the original
utility.


## Process

In other words, DUL accelerates data valuation by learning the utility function
from a small number of subsets. The process is as follows:

1. Collect a given_budget_ of so-called _utility samples_ (subsets and their
   utility values) during the normal course of data valuation.
2. Fit a model $\hat{u}$ to the utility samples. The model is trained to predict
   the utility of new subsets.
3. Continue the valuation process, sampling subsets, but instead of evaluating the
   original utility function, use the learned model to predict it.

## Usage

There are three components (sorry for the confusing naming!):

1. The original utility object to learn, typically (but not necessarily) a
   [ModelUtility][pydvl.valuation.utility.modelutility.ModelUtility] object which will be
   expensive to evaluate. Any subclass of
   [UtilityBase][pydvl.valuation.utility.base.UtilityBase] should work. Let's call
   it `utility`.
2. A [UtilityModel][pydvl.valuation.utility.learning.UtilityModel] which will be
   trained to predict the utility of subsets.
3. The [DataUtilityLearning][pydvl.valuation.utility.learning.DataUtilityLearning]
   object.

Assuming you have some data valuation algorithm and your `utility` object:

1. Pick the actual machine learning model to use to learn the utility. In most
   cases the utility takes continuous values, so this should be any regression
   model, such as a linear regression or a neural network. The input to it will
   be sets of indices, so one has to encode the data accordingly. For example,
   an indicator vector of the set as done in [@wang_improving_2022], with
   [IndicatorUtilityModel][pydvl.valuation.utility.learning.IndicatorUtilityModel].
   This wrapper accepts any machine learning model for the actual fitting.

   An alternative way to encode the data is to use a permutation-invariant model,
   such as [DeepSet][pydvl.valuation.utility.deepset.DeepSet] [@zaheer_deep_2017],
   which is a simple architecture to learn embeddings for sets of points.
2. Wrap both your `utility` object and the utility model just constructed within
   a [DataUtilityLearning][pydvl.valuation.utility.learning.DataUtilityLearning].
3. Use this last object in your data valuation algorithm instead of the original
   `utility`.

## Indicator encoding  { #dul-indicator-encoding-intro }

The authors of DUL propose to use an indicator function to encode the sets of
indices: a vector of length `len(training_data)` with a 1 at index $i$ if sample
$x_i$ is in the set and 0 otherwise. This encoding can then be fed to any
regression model.

While this can work under some circumstances, note that one is effectively
learning a regression function on the corners of an $n$-dimensional hypercube,
a problem well known to be difficult. For this reason, we offer a (naive)
implementation of a permutation-invariant model called [Deep
Sets][deep-sets-intro] which can serve as guidance for a more complex
architecture.

!!! example "DUL with a linear regression model"
    ??? Example
        ``` python
        from pydvl.valuation import Dataset, DataUtilityLearning, ModelUtility, \
            Sample, SupervisedScorer
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.datasets import load_iris

        train, test = Dataset.from_sklearn(load_iris())
        scorer = SupervisedScorer("accuracy", test, 0, (0,1))
        utility = ModelUtility(LinearRegression(), scorer)
        utility_model = IndicatorUtilityModel(LinearRegression(), len(train))
        dul = DataUtilityLearning(utility, 300, utility_model)
        valuation = ShapleyValuation(
            utility=dul,
            sampler=PermutationSampler(),
            stopping=MaxUpdates(6000)
        )
        # Note: DUL does not support parallel training yet
        valuation.fit(train)
        ```

## Deep Sets  { #deep-sets-intro }

Given a set $S= \{x_1, x_2, ..., x_n\},$ Deep Sets [@zaheer_deep_2017] learn a
representation of the set which is invariant to the order of elements in the
set. The model consists of two networks:

$$ \Phi(S) = \sum_{x_i \in S} \phi(x_i), $$

where $\phi(x_i)$ is a learned embedding for data point $x_i,$ and a second network
$\rho$ that predicts the output $y$ from the aggregated representation:

$$ y = \rho(\Phi(S)). $$


!!! example "DUL with DeepSets"
    ??? Example
        This example requires pytorch installed.
        ``` python
        from pydvl.valuation import Dataset, DataUtilityLearning, ModelUtility, \
            Sample, SupervisedScorer
        from pydvl.valuation.utility.deepset import DeepSet
        from sklearn.datasets import load_iris

        train, test = Dataset.from_sklearn(load_iris())
        scorer = SupervisedScorer("accuracy", test, 0, (0,1))
        utility = ModelUtility(LinearRegression(), scorer)
        utility_model = DeepSet(
            input_dim=len(train),
            phi_hidden_dim=10,
            phi_output_dim=20,
            rho_hidden_dim=10
        )
        dul = DataUtilityLearning(utility, 3000, utility_model)

        valuation = ShapleyValuation(
            utility=dul,
            sampler=PermutationSampler(),
            stopping=MaxUpdates(10000)
        )
        # Note: DUL does not support parallel training yet
        valuation.fit(train)
        ```

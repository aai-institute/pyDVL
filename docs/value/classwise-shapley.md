---
title: Class-wise Shapley
alias: classwise-shapley-intro
---

# Class-wise Shapley { #classwise-shapley-intro }

Class-wise Shapley (CWS) [@schoch_csshapley_2022] offers a Shapley framework
tailored for classification problems.  Given a sample $x_i$ with label $y_i \in
\mathbb{N}$, let $D_{y_i}$ be the subset of $D$ with labels $y_i$, and
$D_{-y_i}$ be the complement of $D_{y_i}$ in $D$. The key idea is that the
sample $(x_i, y_i)$ might improve the overall model performance on $D$, while
being detrimental for the performance on $D_{y_i},$ e.g. because of a wrong
label. To address this issue, the authors introduced

$$
v_u(i) = \frac{1}{2^{|D_{-y_i}|}} \sum_{S_{-y_i}}
\frac{1}{|D_{y_i}|}\sum_{S_{y_i}} \binom{|D_{y_i}|-1}{|S_{y_i}|}^{-1}
\delta(S_{y_i} | S_{-y_i}),
$$

where $S_{y_i} \subseteq D_{y_i} \setminus \{i\}$ and $S_{-y_i} \subseteq
D_{-y_i}$ is _arbitrary_ (in particular, not the complement of $S_{y_i}$). The
function $\delta$ is called **set-conditional marginal Shapley value** and is
defined as

$$
\delta(S | C) = u( S_{+i} | C ) − u(S | C),
$$

for any set $S$ such that $i \notin S, C$ and $S \cap C = \emptyset$.

In practical applications, estimating this quantity is done both with Monte
Carlo sampling of the powerset, and the set of index permutations
[@castro_polynomial_2009]. Typically, this requires fewer samples than the
original Shapley value, although the actual speed-up depends on the model and
the dataset.


??? Example "Computing classwise Shapley values"
    CWS is implemented in
    [ClasswiseShapleyValuation][pydvl.valuation.methods.classwise_shapley.ClasswiseShapleyValuation].
    To construct this object the model is passed inside a
    [ClasswiseModelUtility][pydvl.valuation.utility.classwise.ClasswiseModelUtility]
    together with a
    [ClasswiseSupervisedScorer][pydvl.valuation.scorers.classwise.ClasswiseSupervisedScorer]
    The two samplers required by the method are wrapped by a
    [ClasswiseSampler][pydvl.valuation.samplers.classwise.ClasswiseSampler].

    The following example illustrates how to replicate the algorithm in Appendix
    A of [@schoch_csshapley_2022].
    ```python
    from pydvl.valuation import *
    
    seed = 42
    model = ...
    train, test = Dataset.from_arrays(X, y, train_size=0.6, random_state=seed)
    n_labels = len(get_unique_labels(train.data().y))
    scorer = ClasswiseSupervisedScorer("accuracy", test)
    utility = ClasswiseModelUtility(model, scorer)
    sampler = ClasswiseSampler(
        in_class=PermutationSampler(
            truncation=RelativeTruncation(rtol=0.01, burn_in_fraction=0.3), seed=seed
        ),
        out_of_class=UniformSampler(index_iteration=NoIndexIteration),
        max_in_class_samples=1,
    )
    # 500 permutations per label as in the paper
    stopping = MaxSamples(sampler, 500*n_labels)
    # Save the history in valuation.stopping.criteria[1]
    stopping |= History(n_steps=5000),
    valuation = ClasswiseShapleyValuation(
        utility=utility, sampler=sampler, is_done=stopping, normalize_values=True
    )
    ```


### The class-wise scorer

In order to use the class-wise Shapley value, one needs to instantiate a
[ClasswiseSupervisedScorer][pydvl.valuation.scorers.classwise.ClasswiseSupervisedScorer].
This scorer is defined as

$$ u(S) = f(a_S(D_{y_i})) \ g(a_S(D_{-y_i})), $$

where $f$ and $g$ are monotonically increasing functions, $a_S(D_{y_i})$ is the
**in-class accuracy**, and $a_S(D_{-y_i})$ is the **out-of-class accuracy** (the
names originate from a choice by the authors to use accuracy, but in principle
any other score, like $F_1$ can be used). 

The authors show that $f(x)=x$ and $g(x)=e^x$ have favorable properties and are
therefore the defaults, but we leave the option to set different functions $f$
and $g$ for an exploration with different base scores. 

??? example "The default class-wise scorer"
    The CWS scorer requires choosing a metric and the functions $f$ and $g,$
    which by default are set to the values in the paper:

    ```python
    import numpy as np
    from pydvl.valuation.scorers.classwise import ClasswiseSupervisedScorer
    
    _, test = Dataset.from_sklearn(...)
    identity = lambda x: x
    scorer = ClasswiseSupervisedScorer(
        "accuracy",
        default=0.0,
        range=(0.0, 1.0),
        test_data=test,
        in_class_discount_fn=identity,
        out_of_class_discount_fn=np.exp
    )
    ```

??? "Surface of the discounted utility function"
    The level curves for $f(x)=x$ and $g(x)=e^x$ are depicted below. The lines
    illustrate the contour lines, annotated with their respective gradients.
    ![Level curves of the class-wise
    utility](img/classwise-shapley-discounted-utility-function.svg){ align=left width=33%  class=invertible }

## Evaluation

We illustrate the method with two experiments: point removal and noise removal,
as well as an analysis of the distribution of the values. For this we employ the
nine datasets used in [@schoch_csshapley_2022], using the same pre-processing.
For images, PCA is used to reduce down to 32 the features found by a pre-trained
`Resnet18` model. Standard loc-scale normalization is performed for all models
except gradient boosting, since the latter is not sensitive to the scale of the
features.

??? info "Datasets used for evaluation"
    | Dataset        | Data Type | Classes | Input Dims | OpenML ID |
    |----------------|-----------|---------|------------|-----------|
    | Diabetes       | Tabular   | 2       | 8          | 37        |
    | Click          | Tabular   | 2       | 11         | 1216      |
    | CPU            | Tabular   | 2       | 21         | 197       |
    | Covertype      | Tabular   | 7       | 54         | 1596      |
    | Phoneme        | Tabular   | 2       | 5          | 1489      |
    | FMNIST         | Image     | 2       | 32         | 40996     |
    | CIFAR10        | Image     | 2       | 32         | 40927     |
    | MNIST (binary) | Image     | 2       | 32         | 554       |
    | MNIST (multi)  | Image     | 10      | 32         | 554       |

We show mean and coefficient of variation (CV) $\frac{\sigma}{\mu}$ of an "inner
metric". The former shows the performance of the method, whereas the latter
displays its stability: we normalize by the mean to see the relative effect of
the standard deviation. Ideally the mean value is maximal and CV minimal. 

Finally, we note that for all sampling-based valuation methods the same number
of _evaluations of the marginal utility_ was used. This is important to make the
algorithms comparable, but in practice one should consider using a more
sophisticated stopping criterion.

### Dataset pruning for logistic regression (point removal)

In (best-)point removal, one first computes values for the training set and then
removes in sequence the points with the highest values. After each removal, the
remaining points are used to train the model from scratch and performance is
measured on a test set. This produces a curve of performance vs. number of
points removed which we show below.

As a scalar summary of this curve, [@schoch_csshapley_2022] define **Weighted
Accuracy Drop** (WAD) as:

$$
\begin{aligned}
\text{wad} &= \sum_{j=1}^{n} \frac{1}{j} \sum_{i=1}^{j}
  \left( a_{T_{-\{1 : i-1 \}}}(D) - a_{T_{-\{1 : i \}}}(D) \right) \\
   &= a_T(D) - \sum_{j=1}^{n} \frac{a_{T_{-\{1 : j \}}}(D)}{j} ,
\end{aligned}
$$


where $a_T(D)$ is the accuracy of the model (trained on $T$) evaluated on $D$
and $T_{-\{1 : j \}}$ is the set $T$ without elements from $\{1, \dots , j
\}$.

We run the point removal experiment for a logistic regression model five times
and compute WAD for each run, then report the mean $\mu_\text{wad}$ and standard
deviation $\sigma_\text{wad}$.

![Mean WAD for best-point removal on logistic regression. Values
computed using LOO, CWS, Beta Shapley, and TMCS
](img/classwise-shapley-metric-wad-mean.svg){ class=invertible }

We see that CWS is competitive with all three other methods. In all problems
except `MNIST (multi)` it outperforms TMCS, while in that case TMCS has a slight
advantage.

In order to understand the variability of WAD we look at its coefficient of
variation (lower is better):

![Coefficient of Variation of WAD for best-point removal on logistic regression.
Values computed using LOO, CWS, Beta Shapley, and TMCS
](img/classwise-shapley-metric-wad-cv.svg){ class=invertible }

CWS is not the best method in terms of CV. For `CIFAR10`, `Click`, `CPU` and
`MNIST (binary)` Beta Shapley has the lowest CV. For `Diabetes`, `MNIST (multi)`
and `Phoneme` CWS is the winner and for `FMNIST` and `Covertype` TMCS takes the
lead. Besides LOO, TMCS has the highest relative standard deviation.

The following plot shows accuracy vs number of samples removed. Random values
serve as a baseline. The shaded area represents the 95% bootstrap confidence
interval of the mean across 5 runs.

![Accuracy after best-sample removal using values from logistic 
regression](img/classwise-shapley-weighted-accuracy-drop-logistic-regression-to-logistic-regression.svg){ class=invertible }

Because samples are removed from high to low valuation order, we expect a steep
decrease in the curve.

Overall we conclude that in terms of mean WAD, CWS and TMCS perform best, with
CWS's CV on par with Beta Shapley's, making CWS a competitive method.


### Dataset pruning for a neural network by value transfer

Transfer of values from one model to another is probably of greater practical
relevance: values are computed using a cheap model and used to prune the dataset
before training a more expensive one.

The following plot shows accuracy vs number of samples removed for transfer from
logistic regression to a neural network. The shaded area represents the 95%
bootstrap confidence interval of the mean across 5 runs.

![Accuracy after sample removal using values transferred from logistic
regression to an MLP
](img/classwise-shapley-weighted-accuracy-drop-logistic-regression-to-mlp.svg){ class=invertible }

As in the previous experiment samples are removed from high to low valuation
order and hence we expect a steep decrease in the curve. CWS is competitive with
the other methods, especially in very unbalanced datasets like `Click`. In other
datasets, like `Covertype`, `Diabetes` and `MNIST (multi)` the performance is on
par with TMCS.


### Detection of mis-labeled data points

The next experiment tries to detect mis-labeled data points in binary
classification tasks. 20% of the indices is flipped at random (we don't consider
multi-class datasets because there isn't a unique flipping strategy). The
following table shows the mean of the area under the curve (AUC) for five runs.

![Mean AUC for mis-labeled data point detection. Values computed using LOO, CWS,
Beta Shapley, and 
TMCS](img/classwise-shapley-metric-auc-mean.svg){ class=invertible }

In the majority of cases TMCS has a slight advantage over CWS, except for
`Click`, where CWS has a slight edge, most probably due to the unbalanced nature
of the dataset. The following plot shows the CV for the AUC of the five runs.

![Coefficient of variation of AUC for mis-labeled data point detection. Values
computed using LOO, CWS, Beta Shapley, and TMCS
](img/classwise-shapley-metric-auc-cv.svg){ class=invertible }
 
In terms of CV, CWS has a clear edge over TMCS and Beta Shapley.

Finally, we look at the ROC curves training the classifier on the $n$ first
samples in _increasing_ order of valuation (i.e. starting with the worst):

![Mean ROC across 5 runs with 95% bootstrap
CI](img/classwise-shapley-roc-auc-logistic-regression.svg){ class=invertible }

Although at first sight TMCS seems to be the winner, CWS stays competitive after
factoring in running time. For a perfectly balanced dataset, CWS needs on
average fewer samples than TCMS.

### Value distribution

For illustration, we compare the distribution of values computed by TMCS and
CWS.

![Histogram and estimated density of the values computed by TMCS and
CWS on all nine datasets](img/classwise-shapley-density.svg){ class=invertible }

For `Click` TMCS has a multi-modal distribution of values. We hypothesize that
this is due to the highly unbalanced nature of the dataset, and notice that CWS
has a single mode, leading to its greater performance on this dataset.

## Conclusion

CWS is an effective way to handle classification problems, in particular for
unbalanced datasets. It reduces the computing requirements by considering
in-class and out-of-class points separately.


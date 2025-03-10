---
title: Delta-Shapley
alias: 
  name: delta-shapley-intro
  text: Introduction to Delta-Shapley
---

# $\delta$-Shapley { #delta-shapley-intro }

[$\delta$-Shapley][pydvl.valuation.methods.delta_shapley.DeltaShapleyValuation]
is a semi-value that employs a constant sampling probability, truncated for sets
beyond a certain range. Through some error analysis, it is shown that for
certain model classes, small coalitions are good estimators of the data value.
This is mainly due to the fact that adding more data has diminishing returns for
many models.[@watson_accelerated_2023]

The value of a point $i$ is defined as:

$$
v_\delta(i) = \sum_{k=l}^u w(k) \sum_{S \subset D_{-i}^{(k)}} [U(S_{+i}) - U(S)],
$$

where $l$ and $u$ are the lower and upper bounds of the size of the subsets to
sample from, and $w(k)$ is the weight of a subset of size $k$ in the complement
of $\{i\}$, and is given by:

$$
\begin{array}{ll}
w (k) = \left \{
    \begin{array}{ll}
        \frac{1}{u - l + 1} & \text{if} \ l \ \leq k \leq u,\\ 0 &
        \text{otherwise.}
    \end{array} \right. &
    \end{array}
$$

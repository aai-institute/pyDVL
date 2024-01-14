---
title: Notation for valuation
---

# Notation for valuation

!!! todo
    Organize this page better and use its content consistently throughout the
    documentation.

The following notation is used throughout the documentation:

Let $D = \{x_1, \ldots, x_n\}$ be a training set of $n$ samples.

The utility function $u:\mathcal{D} \rightarrow \mathbb{R}$ maps subsets of $D$
to real numbers. In pyDVL, we typically call this mappin a **score** for
consistency with sklearn, and reserve the term **utility** for the triple of
dataset $D$, model $f$ and score $u$, since they are used together to compute
the value.

The value $v$ of the $i$-th sample in dataset $D$ wrt. utility $u$ is
denoted as $v_u(x_i)$ or simply $v(i)$.

For any $S \subseteq D$, we denote by $S_{-i}$ the set of samples in $D$
excluding $x_i$, and $S_{+i}$ denotes the set $S$ with $x_i$ added.

The marginal utility of adding sample $x_i$ to a subset $S$ is denoted as
$\delta(i) := u(S_{+i}) - u(S)$.

The set $D_{-i}^{(k)}$ contains all subsets of $D$ of size $k$ that do not
include sample $x_i$.

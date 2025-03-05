---
title: Mixing sampling strategies for semi-values
alias:
  name: semi-values-sampling
---

# Mixing sampling strategies for semi-values

This document explains the rationale behind the product of coefficients for the
sampling strategy and the method in the semi-value methods. Samplers define
so-called _sampling strategies_, which subclass
[EvaluationStrategy][pydvl.valuation.samplers.EvaluationStrategy]. These
strategies are where sub-processes compute marginals, or incrementally iterate
over permutations, or perform whichever computation is required depending on the
sampling scheme. Crucially, they compute **the product of a semi-value
coefficient with a sampling weight**. Here is why. 

??? example "The core semi-value computation for powerset sampling"
    This is the core of the marginal update computation in
    [PowersetEvaluationStrategy][pydvl.valuation.samplers.powerset.PowersetEvaluationStrategy]:
    
    ```python
    for sample in batch:
        u_i = self.utility(sample.with_idx_in_subset())
        u = self.utility(sample)
        marginal = u_i - u
        sign = np.sign(marginal)
        log_marginal = -np.inf if marginal == 0 else np.log(marginal * sign)
        log_marginal += self.log_correction(self.n_indices, len(sample.subset))
        updates.append(ValueUpdate(sample.idx, log_marginal, sign))
    ```  

## Correction weights for Shapley values

###  Uniform sampling over subsets

We begin by rewriting the combinatorial definition of Shapley value:

$$
\begin{eqnarray*}
  v_{\text{sh}} (i) & = & \frac{1}{N}  \sum_{S \subseteq D_{- i}}
  \binom{N - 1}{| S |}^{- 1}  [U (S_{+ i}) - U (S)],\\\
  & = & \sum_{S \subseteq D_{- i}} w_{\text{sh}} (| S |) \delta
  _{i} (S),
\end{eqnarray*}
$$

where $\delta _{i} (S) := U (S_{+ i}) - U (S).$

The natural Monte Carlo approximation is then to sample $S_{j} \sim 
\mathcal{U} (D_{- i})$ and let

$$
\begin{eqnarray*}
  \hat{v}_{\text{sh}, \text{unif}} (i) & = & \frac{2^{N -
  1}}{m}  \sum_{j = 1}^m w_{\text{sh}} (| S_{j} |) \delta _{i}
  (S_{j})\\\
  & \underset{m \rightarrow \infty}{\longrightarrow} & \underset{S \sim
  \mathcal{U} (D_{- i})}{\mathbb{E}} [w_{\text{sh}} (| S |) \delta
  _{i} (S)] 2^{N - 1}\\\
  & = & \sum_{S \subseteq D_{- i}} w_{\text{sh}} (| S |) \delta
  _{i} (S)
\end{eqnarray*}
$$

So, because of the sampling strategy we chose, we had to add a coefficient 
$2^{N - 1}$ in order to recover the value of $v_{i}$ in the limit. We can call 
this coefficient $w_{\text{unif}} := 2^{N - 1}$. At every step of the 
MC algorithm we:

!!! abstract "Algorithm 1"
    1. sample $S_{j}$
    2. compute the marginal
    3. compute the product of coefficients for the sampler and the method 
       $w_{\text{unif}} w_{\text{sh}} (| S_{j} |)$
    4. update the running average for $\hat{v}_{\text{unif}, 
       \text{sh}}$

### The case of permutations

Consider now the alternative choice of uniformly sampling permutations $\sigma 
_{j} \sim \mathcal{U} (\Pi (D))$. Recall that we let $S_{i}^{\sigma}$ be the 
set of indices preceding $i$ in the permutation $\sigma$. Then

$$
\begin{eqnarray*}
  \hat{v}_{\text{sh}, \text{per}} (i) & = & \frac{1}{m}
  \sum_{j = 1}^m w_{\text{per}} (| S_{i}^{\sigma _{j}} |) \delta
  _{i} (S_{i}^{\sigma _{j}})\\\
  & \longrightarrow & \underset{\sigma \sim \mathcal{U} (\Pi (D))}{\mathbb{E}}
  [w_{\text{per}} (| S_{i}^{\sigma} |) \delta _{i}
  (S_{i}^{\sigma})]\\\
  & = & \frac{1}{N!}  \sum_{\sigma \in \Pi (D)} w_{\text{per}} (|
  S_{i}^{\sigma} |)  [U (S_{i}^{\sigma} \cup \lbrace i \rbrace) - U
  (S_{i}^{\sigma})]\\\
  & \overset{(\star)}{=} & \frac{1}{N!}  \sum_{S \subset D_{- i}}
  w_{\text{per}} (| S |)  (N - 1 - | S |) ! | S | ! [U (S_{+ i}) - U
  (S)]\\\
  & = & \frac{1}{N}  \sum_{S \subset D_{- i}} w_{\text{per}} (| S
  |)  \binom{N - 1}{| S |}^{- 1} \delta _{i} (S)\\\
  & = & \sum_{S \subset D_{- i}} w_{\text{per}} (| S |)
  w_{\text{sh}} (| S |) \delta _{i} (S)\\\
  & = & v_{\text{sh}} (i) \text{, if } w_{\text{per}} \equiv
  1,
\end{eqnarray*}
$$

So we see that for permutation sampling we can choose a constant coefficient of 
1, and simply compute an average of marginals, and we will recover 
$v_{\text{sh}}$. At step $(\star)$ we have counted the number of 
permutations before a fixed position of index $i$ and after it, because the 
utility does not depend on ordering. This allows us to sum over sets instead of 
permutations.

However, for consistency with the previous algorithm, we will choose 
$w_{\text{per}} (k) = w_{\text{sh}} (k)^{- 1}$ and include a 
term $w_{\text{sh}}$ in the averages, to cancel it out:[^1]

!!! abstract "Algorithm 2"
    1. sample $\sigma _{j}$
    2. compute the marginal $\delta _{i} (S_{i}^{\sigma _{j}})$
    3. compute the product of coefficients for the sampler and the method 
       $w_{\text{per}} (| S_{i}^{\sigma} |) w_{\text{sh}} (| 
        S_{i}^{\sigma} |) = 1$
    4. update the running average for $\hat{v}_{\text{per}, 
      \text{sh}}$

## General semi-values

Let's look now at general semi-values, which are of the form:

$$
\begin{eqnarray*}
  v_{\text{semi}} (i) & = & \sum_{k = 0}^{n - 1} \tilde{w} (k)
  \sum_{S \subseteq D_{- i}^{(k)}} \delta _{i} (S),
\end{eqnarray*}
$$

where $D^{(k)}_{- i} := \lbrace S \subseteq D_{- i} : | S | = k \rbrace$ and 
$\sum \tilde{w} (k) = 1$. Note that taking $\tilde{w} (k) = 
w_{\text{sh}}$ we arrive at $v_{\text{sh}}$ simply by 
splitting the sum in $v_{\text{sh}}$ by size of subsets $S \subset 
D_{- i}$. Another choice is $w_{\text{bzf}} := 2^{- (N - 1)}$.

To fix ideas let's focus on the Banzhaf semi-value.

Let's sample permutations $\sigma _{j} \sim \mathcal{U} (\Pi (D))$ and 
approximate $v_{\text{bzf}}$:

$$
\begin{eqnarray*}
  \hat{v}_{\text{bzf}, \text{per}} (i) & = & \frac{1}{m}
  \sum_{j = 1}^m w_{\text{bzf}} w_{\text{per}} (|
  S_{i}^{\sigma _{j}} |) \delta _{i} (S_{i}^{\sigma _{j}})\\\
  & \longrightarrow & \underset{\sigma \sim \mathcal{U} (\Pi (D))}{\mathbb{E}}
  [w_{\text{bzf}} w_{\text{per}} (| S_{i}^{\sigma} |) \delta
  _{i} (S_{i}^{\sigma})]\\\
  & = & \sum_{S \subset D_{- i}} w_{\text{bzf}}
  w_{\text{per}} (| S |)  \frac{1}{N}  \binom{N - 1}{| S |}^{- 1}
  \delta _{i} (S)\\\
  & = & \sum_{k = 0}^{n - 1} w_{\text{bzf}} w_{\text{per}}
  (k) w_{\text{sh}} (k)  \sum_{S \subseteq D_{- i}^{(k)}} \delta
  _{i} (S),\\\
  & = & \sum_{k = 0}^{n - 1} w_{\text{bzf}}  \sum_{S \subseteq
  D_{- i}^{(k)}} \delta _{i} (S), \text{ if } w_{\text{per}} = 1 /
  w_{\text{sh}}
\end{eqnarray*}
$$

so we see that we must have $w_{\text{per}} (k) = 1 / 
w_{\text{sh}} (k)$ in order to recover $v_{\text{bzf}} (i)$.

For uniform sampling of the powerset $S \sim \mathcal{U} (D_{- i})$, we do as 
above, and, because $w_{\text{unif}} w_{\text{bzf}} = 1$:

$$
\begin{eqnarray*}
  \hat{v}_{\text{bzf}, \text{unif}} (i) & := & \frac{1}{m}
  \sum_{j = 1}^m w_{\text{unif}} w_{\text{bzf}} \delta _{i}
  (S_{j})\\\
  & \longrightarrow & \underset{S \sim \mathcal{U} (D_{- i})}{\mathbb{E}}
  [\delta _{i} (S)]\\\
  & = & v_{\text{bzf}} (i) .
\end{eqnarray*}
$$

## Conclusion

We have a general way of mixing sampling strategies and semi-value coefficients.
The drawback is that a direct implementation with that much cancelling of
coefficients might be inefficient or numerically unstable. On the flip side, we
can implement any sampling method, like antithetic sampling, and immediately
benefit in all semi-value computations.

!!! Danger "Numerical instability"
    For some sampling schemes the instability can be severe. For example, the
    [Maximum Sample Reuse][pydvl.valuation.samplers.msr.MSRSampler] sampler
    introduced in [@wang_data_2023] is very efficient for Banzhaf indices, but
    when used with Shapley values it can be very unreliable.
  

[^1]: From a numerical point of view, this is a bad idea, since it means
computing very large numbers and then dividing one by the other. However, it
makes the implementation very homogeneous. To alleviate the issue we perform
all computation in log-space and use the log-sum-exponent trick to compute
running quantities.

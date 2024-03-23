---
title: Methods
alias: 
  name: methods
  text: Methods
---

We currently implement the following methods:

## Data Valuation

- [**LOO**][pydvl.value.loo.compute_loo].

- [**Permutation Shapley**][pydvl.value.shapley.montecarlo.permutation_montecarlo_shapley]
  (also called **ApproxShapley**) [@castro_polynomial_2009].

- [**TMCS**][pydvl.value.shapley.compute_shapley_values]
  [@ghorbani_data_2019].

- [**Data Banzhaf**][pydvl.value.semivalues.compute_banzhaf_semivalues]
  [@wang_data_2022].

- [**Beta Shapley**][pydvl.value.semivalues.compute_beta_shapley_semivalues]
  [@kwon_beta_2022].

- [**CS-Shapley**][pydvl.value.shapley.classwise.compute_classwise_shapley_values]
  [@schoch_csshapley_2022].

- [**Least Core**][pydvl.value.least_core.montecarlo.montecarlo_least_core]
  [@yan_if_2021].

- [**Owen Sampling**][pydvl.value.shapley.owen.owen_sampling_shapley]
  [@okhrati_multilinear_2021].

- [**Data Utility Learning**][pydvl.utils.utility.DataUtilityLearning]
  [@wang_improving_2022].

- [**kNN-Shapley**][pydvl.value.shapley.knn.knn_shapley]
  [@jia_efficient_2019a].

- [**Group Testing**][pydvl.value.shapley.gt.group_testing_shapley]
  [@jia_efficient_2019]

- [**Data-OOB**][pydvl.value.oob.compute_data_oob]
  [@kwon_dataoob_2023].

## Influence Functions

- [**CG Influence**][pydvl.influence.torch.CgInfluence].
  [@koh_understanding_2017].

- [**Direct Influence**][pydvl.influence.torch.DirectInfluence]
  [@koh_understanding_2017].

- [**LiSSA**][pydvl.influence.torch.LissaInfluence]
  [@agarwal_secondorder_2017].

- [**Arnoldi Influence**][pydvl.influence.torch.ArnoldiInfluence]
  [@schioppa_scaling_2021].

- [**EKFAC Influence**][pydvl.influence.torch.EkfacInfluence]
  [@george_fast_2018;@martens_optimizing_2015].

- [**Nyström Influence**][pydvl.influence.torch.NystroemSketchInfluence], based
  on the ideas in [@hataya_nystrom_2023] for bi-level optimization.

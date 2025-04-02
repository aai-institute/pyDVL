---
title: Methods
alias: 
  name: methods
  text: Methods
---

We currently implement the following methods:

## Data valuation  { #implemented-methods-data-valuation }

- [$\delta$-Shapley][delta-shapley-intro]
  [@watson_accelerated_2023]

- [Beta Shapley][beta-shapley-intro]
  [@kwon_beta_2022].

- [Class-Wise Shapley][classwise-shapley-intro]
  [@schoch_csshapley_2022].

- [Data Banzhaf][data-banzhaf-intro] and [MSR sampling][msr-banzhaf-intro]
  [@wang_data_2023].

- [Data Utility Learning][data-utility-learning-intro]
  [@wang_improving_2022].

- [Data-OOB][data-oob-intro]
  [@kwon_dataoob_2023].

- [Group Testing Shapley][group-testing-shapley-intro]
  [@jia_efficient_2019]

- [kNN-Shapley][knn-shapley-intro], exact only
  [@jia_efficient_2019a].

- [Least Core][least-core-intro]
  [@yan_if_2021].

- [Leave-One-Out values][loo-valuation-intro].

- [Owen Shapley][owen-shapley-intro]
  [@okhrati_multilinear_2021].

- [Permutation Shapley][permutation-shapley-intro], also called _ApproxShapley_
  [@castro_polynomial_2009].

- [Truncated Monte Carlo Shapley][tmcs-intro]
  [@ghorbani_data_2019].

- [Variance-Reduced Data Shapley][pydvl.valuation.samplers.stratified.VRDSSampler]
  [@wu_variance_2023].

## Influence functions  { #implemented-methods-influence-functions }

- [CG Influence][pydvl.influence.torch.CgInfluence]
  [@koh_understanding_2017].

- [Direct Influence][pydvl.influence.torch.DirectInfluence]
  [@koh_understanding_2017].

- [LiSSA][pydvl.influence.torch.LissaInfluence]
  [@agarwal_secondorder_2017].

- [Arnoldi Influence][pydvl.influence.torch.ArnoldiInfluence]
  [@schioppa_scaling_2022].

- [EKFAC Influence][pydvl.influence.torch.EkfacInfluence]
  [@george_fast_2018;@martens_optimizing_2015].

- [Nyström Influence][pydvl.influence.torch.NystroemSketchInfluence], based
  on the ideas in [@hataya_nystrom_2023] for bi-level optimization.

- [Inverse-harmonic-mean
  Influence][pydvl.influence.torch.InverseHarmonicMeanInfluence]
  [@kwon_datainf_2023].


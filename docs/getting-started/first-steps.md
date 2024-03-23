---
title: First Steps
alias: 
  name: first-steps
  text: First Steps
---

# First Steps

!!! Warning
    Make sure you have read [[getting-started#installation]] before using the library. 
    In particular read about which extra dependencies you may need.

## Main Concepts

pyDVL aims to be a repository of production-ready, reference implementations of
algorithms for data valuation and influence functions. Even though we only
briefly introduce key concepts in the documentation, the following sections 
should be enough to get you started.

* [[data-valuation]] for key objects and usage patterns for Shapley value
  computation and related methods.
* [[influence-function]] for instructions on how to compute influence functions.

## Running the Examples

If you are somewhat familiar with the concepts of data valuation, you can start
by browsing our worked-out examples illustrating pyDVL's capabilities either:

- In the examples under [[data-valuation]] and [[influence-function]].
- Using [binder](https://mybinder.org/) notebooks, deployed from each
  example's page.
- Locally, by starting a jupyter server at the root of the project. You will
  have to install jupyter first manually since it's not a dependency of the
  library.

## Advanced Usage

Refer to the [[advanced-usage]] page for explanations on how to enable
and use parallelization and caching.

---
title: First steps
alias: 
  name: first-steps
  text: First Steps
---

# First steps

!!! info
    Make sure you have read [[getting-started#installation]] before using the
    library. In particular read about which extra dependencies you may need.

## Main concepts

pyDVL aims to be a repository of production-ready, reference implementations of
algorithms for data valuation and influence functions. Read the following
sections to get started:

<div class="grid cards" markdown>

-   [[data-valuation-intro]]
  
    ---
    Key objects and usage patterns for Shapley values and related methods.
  
    [[data-valuation-intro|:octicons-arrow-right-24: Data valuation]]
  
-   [[influence-function]]
  
    ---
    Instructions on how to compute influence functions, and many approximations.
    
    [[influence-function|:octicons-arrow-right-24: Influence functions]]

</div>

## Supported frameworks

* The module for influence functions is built around PyTorch. Because of our use
  of the `torch.func` stateless api, we do not support jitted modules yet (see
  [#640](https://github.com/aai-institute/pyDVL/issues/640)).

* Up until v0.10.0, pyDVL only supported NumPy arrays for data valuation. From
  version 0.10.1 onwards, the library also supports PyTorch tensors for most
  valuation methods. The implementation attempts to preserve the input data type
  for the [Dataset][pydvl.valuation.dataset.Dataset] throughout computations where
  possible.

  Note that some features have specific requirements or limitations when using
  tensors. For details on tensor support and caveats, see the [[tensor-support]]
  section.


## Running the examples

If you are somewhat familiar with the concepts of data valuation, you can start
by browsing our worked-out examples illustrating pyDVL's capabilities either:

- In the examples under [[data-valuation-intro]] and [[influence-function]].
- Using [binder](https://mybinder.org/) notebooks, deployed from each
  example's page.
- Locally, by starting a jupyter server at the root of the project. You will
  have to install jupyter first manually since it's not a dependency of the
  library.

## Advanced usage

Refer to the [[advanced-usage]] page for explanations on how to enable
and use parallelization and caching.

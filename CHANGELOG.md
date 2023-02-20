# Changelog

## Unreleased

- Added `Scorer` class for a cleaner interface. Fix minor bugs around
  Group-Testing Shapley and switched to cvxpy for the constraint solver.
  [PR #264](https://github.com/appliedAI-Initiative/pyDVL/pull/264)
- Generalised stopping criteria for valuation algorithms. Improved classes
  `ValuationResult` and `Status` with more operations. Some minor issues fixed.
  [PR #252](https://github.com/appliedAI-Initiative/pyDVL/pull/250)
- Fixed a bug whereby `compute_shapley_values` would only spawn one process when
  using `n_jobs=-1` and Monte Carlo methods.
  [PR #270](https://github.com/appliedAI-Initiative/pyDVL/pull/270)
- Bugfix in `RayParallelBackend`: wrong semantics for `kwargs`.
  [PR #268](https://github.com/appliedAI-Initiative/pyDVL/pull/268)
- Splitting of problem preparation and solution in Least-Core computation.
  Umbrella function for LC methods.
  [PR #257](https://github.com/appliedAI-Initiative/pyDVL/pull/257) 
- Operations on `ValuationResult` and `Status` and some cleanup
  [PR #248](https://github.com/appliedAI-Initiative/pyDVL/pull/248)
- **Bug fix and minor improvements**: Fixes bug in TMCS with remote Ray cluster,
  raises an error for dummy sequential parallel backend with TMCS, clones model
  inside `Utility` before fitting by default, with flag `clone_before_fit` 
  to disable it, catches all warnings in `Utility` when `show_warnings` is 
  `False`. Adds Miner and Gloves toy games utilities
  [PR #247](https://github.com/appliedAI-Initiative/pyDVL/pull/247)

## 0.4.0 - 🏭💥 New algorithms and more breaking changes

- GH action to mark issues as stale
  [PR #201](https://github.com/appliedAI-Initiative/pyDVL/pull/201)
- Disabled caching of Utility values as well as repeated evaluations by default
  [PR #211](https://github.com/appliedAI-Initiative/pyDVL/pull/211)
- Test and officially support Python version 3.9 and 3.10 
  [PR #208](https://github.com/appliedAI-Initiative/pyDVL/pull/208)
- **Breaking change:** Introduces a class ValuationResult to gather and inspect
  results from all valuation algorithms
  [PR #214](https://github.com/appliedAI-Initiative/pyDVL/pull/214)
- Fixes bug in Influence calculation with multidimensional input and adds new
  example notebook
  [PR #195](https://github.com/appliedAI-Initiative/pyDVL/pull/195)
- **Breaking change**: Passes the input to `MapReduceJob` at initialization,
  removes `chunkify_inputs` argument from `MapReduceJob`, removes `n_runs`
  argument from `MapReduceJob`, calls the parallel backend's `put()` method for
  each generated chunk in `_chunkify()`, renames ParallelConfig's `num_workers`
  attribute to `n_local_workers`, fixes a bug in `MapReduceJob`'s chunkification
  when `n_runs` >= `n_jobs`, and defines a sequential parallel backend to run
  all jobs in the current thread
  [PR #232](https://github.com/appliedAI-Initiative/pyDVL/pull/232)
- **New method**: Implements exact and monte carlo Least Core for data valuation,
  adds `from_arrays()` class method to the `Dataset` and `GroupedDataset`
  classes, adds `extra_values` argument to `ValuationResult`, adds
  `compute_removal_score()` and `compute_random_removal_score()` helper functions
  [PR #237](https://github.com/appliedAI-Initiative/pyDVL/pull/237)
- **New method**: Group Testing Shapley for valuation, from _Jia et al. 2019_
  [PR #240](https://github.com/appliedAI-Initiative/pyDVL/pull/240)
- Fixes bug in ray initialization in `RayParallelBackend` class
  [PR #239](https://github.com/appliedAI-Initiative/pyDVL/pull/239)
- Implements "Egalitarian Least Core", adds [cvxpy](https://www.cvxpy.org/) as a
  dependency and uses it instead of scipy as optimizer
  [PR #243](https://github.com/appliedAI-Initiative/pyDVL/pull/243)

## 0.3.0 - 💥 Breaking changes

- Simplified and fixed powerset sampling and testing
  [PR #181](https://github.com/appliedAI-Initiative/pyDVL/pull/181)
- Simplified and fixed publishing to PyPI from CI
  [PR #183](https://github.com/appliedAI-Initiative/pyDVL/pull/183)
- Fixed bug in release script and updated contributing docs.
  [PR #184](https://github.com/appliedAI-Initiative/pyDVL/pull/184)
- Added Pull Request template
  [PR #185](https://github.com/appliedAI-Initiative/pyDVL/pull/185)
- Modified Pull Request template to automatically link PR to issue
  [PR ##186](https://github.com/appliedAI-Initiative/pyDVL/pull/186)
- First implementation of Owen Sampling, squashed scores, better testing
  [PR #194](https://github.com/appliedAI-Initiative/pyDVL/pull/194)
- Improved documentation on caching, Shapley, caveats of values, bibtex
  [PR #194](https://github.com/appliedAI-Initiative/pyDVL/pull/194)
- **Breaking change:** Rearranging of modules to accommodate for new methods
  [PR #194](https://github.com/appliedAI-Initiative/pyDVL/pull/194)


## 0.2.0 - 📚 Better docs

Mostly API documentation and notebooks, plus some bugfixes.

### Added

In [PR #161](https://github.com/appliedAI-Initiative/pyDVL/pull/161):
- Support for $$ math in sphinx docs.
- Usage of sphinx extension for external links (introducing new directives like
  `:gh:`, `:issue:` and `:tfl:` to construct standardised links to external
  resources).
- Only update auto-generated documentation files if there are changes. Some
  minor additions to `update_docs.py`.
- Parallelization of exact combinatorial Shapley.
- Integrated KNN shapley into the main interface `compute_shapley_values`.

### Changed

In [PR #161](https://github.com/appliedAI-Initiative/pyDVL/pull/161):
- Improved main docs and Shapley notebooks. Added or fixed many docstrings,
  readme and documentation for contributors. Typos, grammar and style in code,
  documentation and notebooks.
- Internal renaming and rearranging in the parallelization and caching modules.

### Fixed

- Bug in random matrix generation
  [PR #161](https://github.com/appliedAI-Initiative/pyDVL/pull/161).
- Bugs in MapReduceJob's `_chunkify` and `_backpressure` methods
  [PR #176](https://github.com/appliedAI-Initiative/pyDVL/pull/176).


## 0.1.0 - 🎉 first release

This is very first release of pyDVL.

It contains:

- Data Valuation Methods:

  - Leave-One-Out
  - Influence Functions
  - Shapley:
    - Exact Permutation and Combinatorial
    - Montecarlo Permutation and Combinatorial
    - Truncated Montecarlo Permutation
- Caching of results with Memcached
- Parallelization of computations with Ray
- Documentation
- Notebooks containing examples of different use cases

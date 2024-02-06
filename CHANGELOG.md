# Changelog

## Unreleased

### Fixed

- Bug in `LissaInfluence`, when not using CPU device [PR #495](https://github.com/aai-institute/pyDVL/pull/495)

## 0.8.1 - 🆕 🏗  New method and noteboo, Games with exact shapley values, bug fixes and cleanup

### Added

- Implement new method: `EkfacInfluence`
  [PR #451](https://github.com/aai-institute/pyDVL/issues/451)
- New notebook to showcase ekfac for LLMs
  [PR #483](https://github.com/aai-institute/pyDVL/pull/483)
- Implemented exact games in Castro et al. 2009 and 2017
  [PR #341](https://github.com/appliedAI-Initiative/pyDVL/pull/341)

### Fixed

- Bug in using `DaskInfluenceCalcualator` with `TorchnumpyConverter`
  for single dimensional arrays [PR #485](https://github.com/aai-institute/pyDVL/pull/485)
- Fix implementations of `to` methods of `TorchInfluenceFunctionModel` implementations
  [PR #487](https://github.com/aai-institute/pyDVL/pull/487)
- Fixed bug with checking for converged values in semivalues
  [PR #341](https://github.com/appliedAI-Initiative/pyDVL/pull/341)

### Docs

- Add applications of data valuation section, display examples more prominently,
  make all sections visible in table of contents, use mkdocs material cards
  in the home page [PR #492](https://github.com/aai-institute/pyDVL/pull/492)

## 0.8.0 - 🆕 New interfaces, scaling computation, bug fixes and improvements 🎁

### Added

- New cache backends: InMemoryCacheBackend and DiskCacheBackend
  [PR #458](https://github.com/aai-institute/pyDVL/pull/458)
- New influence function interface `InfluenceFunctionModel`
- Data parallel computation with `DaskInfluenceCalculator`
  [PR #26](https://github.com/aai-institute/pyDVL/issues/26)
- Sequential batch-wise computation and write to disk with `SequentialInfluenceCalculator` 
  [PR #377](https://github.com/aai-institute/pyDVL/issues/377)
- Adapt notebooks to new influence abstractions
  [PR #430](https://github.com/aai-institute/pyDVL/issues/430)

### Changed

- Refactor and simplify caching implementation 
  [PR #458](https://github.com/aai-institute/pyDVL/pull/458)
- Simplify display of computation progress
  [PR #466](https://github.com/aai-institute/pyDVL/pull/466)
- Improve readme and explain better the examples
  [PR #465](https://github.com/aai-institute/pyDVL/pull/465)
- Simplify and improve tests, add CodeCov code coverage
  [PR #429](https://github.com/aai-institute/pyDVL/pull/429)
- **Breaking Changes**
  - Removed `compute_influences` and all related code.
    Replaced by new `InfluenceFunctionModel` interface. Removed modules:
    - influence.general
    - influence.inversion
    - influence.twice_differentiable
    - influence.torch.torch_differentiable

### Fixed
- Import bug in README [PR #457](https://github.com/aai-institute/pyDVL/issues/457)

## 0.7.1 - 🆕 New methods, bug fixes and improvements for local tests 🐞🧪

### Added

- New method: Class-wise Shapley values
  [PR #338](https://github.com/aai-institute/pyDVL/pull/338)
- New method: Data-OOB by @BastienZim 
  [PR #426](https://github.com/aai-institute/pyDVL/pull/426), 
  [PR $431](https://github.com/aai-institute/pyDVL/pull/431)
- Added `AntitheticPermutationSampler`
  [PR #439](https://github.com/aai-institute/pyDVL/pull/439)
- Faster semi-value computation with per-index check of stopping criteria (optional)
  [PR #437](https://github.com/aai-institute/pyDVL/pull/437)

### Changed

- No longer using docker within tests to start a memcached server
  [PR #444](https://github.com/aai-institute/pyDVL/pull/444)
- Using pytest-xdist for faster local tests
  [PR #440](https://github.com/aai-institute/pyDVL/pull/440)
- Improvements and fixes to notebooks
  [PR #436](https://github.com/aai-institute/pyDVL/pull/436)
- Refactoring of parallel module. Old imports will stop working in v0.9.0
  [PR #421](https://github.com/aai-institute/pyDVL/pull/421)

### Fixed

- Fix initialization of `data_names` in `ValuationResult.zeros()`
  [PR #443](https://github.com/aai-institute/pyDVL/pull/443)


## 0.7.0 - 📚🆕 Documentation and IF overhaul, new methods and bug fixes 💥🐞

This is our first β release! We have worked hard to deliver improvements across
the board, with a focus on documentation and usability. We have also reworked
the internals of the `influence` module, improved parallelism and handling of
randomness.

### Added

- Implemented solving the Hessian equation via spectral low-rank approximation
  [PR #365](https://github.com/aai-institute/pyDVL/pull/365)
- Enabled parallel computation for Leave-One-Out values
  [PR #406](https://github.com/aai-institute/pyDVL/pull/406)
- Added more abbreviations to documentation
  [PR #415](https://github.com/aai-institute/pyDVL/pull/415)
- Added seed to functions from `pydvl.utils.numeric`, `pydvl.value.shapley` and
  `pydvl.value.semivalues`. Introduced new type `Seed` and conversion function 
  `ensure_seed_sequence`.
  [PR #396](https://github.com/aai-institute/pyDVL/pull/396)
- Added `batch_size` parameter to `compute_banzhaf_semivalues`,
  `compute_beta_shapley_semivalues`, `compute_shapley_semivalues` and
  `compute_generic_semivalues`.
  [PR #428](https://github.com/aai-institute/pyDVL/pull/428)
- Added classwise Shapley as proposed by (Schoch et al. 2021)
  [https://arxiv.org/abs/2211.06800]
  [PR #338](https://github.com/aai-institute/pyDVL/pull/338)

### Changed

- Replaced sphinx with mkdocs for documentation. Major overhaul of documentation
  [PR #352](https://github.com/aai-institute/pyDVL/pull/352)
- Made ray an optional dependency, relying on joblib as default parallel backend
  [PR #408](https://github.com/aai-institute/pyDVL/pull/408)
- Decoupled `ray.init` from `ParallelConfig` 
  [PR #373](https://github.com/aai-institute/pyDVL/pull/383)
- **Breaking Changes**
  - Signature change: return information about Hessian inversion from
    `compute_influence_factors`
    [PR #375](https://github.com/aai-institute/pyDVL/pull/376)
  - Major changes to IF interface and functionality. Foundation for a framework
    abstraction for IF computation.
    [PR #278](https://github.com/aai-institute/pyDVL/pull/278)
    [PR #394](https://github.com/aai-institute/pyDVL/pull/394)
  - Renamed `semivalues` to `compute_generic_semivalues`
    [PR #413](https://github.com/aai-institute/pyDVL/pull/413)
  - New `joblib` backend as default instead of ray. Simplify MapReduceJob.
    [PR #355](https://github.com/aai-institute/pyDVL/pull/355)
  - Bump torch dependency for influence package to 2.0
    [PR #365](https://github.com/aai-institute/pyDVL/pull/365)

### Fixed

- Fixes to parallel computation of generic semi-values: properly handle all
  samplers and stopping criteria, irrespective of parallel backend.
  [PR #372](https://github.com/aai-institute/pyDVL/pull/372)
- Optimises memory usage in IF calculation
  [PR #375](https://github.com/aai-institute/pyDVL/pull/376)
- Fix adding valuation results with overlapping indices and different lengths
  [PR #370](https://github.com/aai-institute/pyDVL/pull/370)
- Fixed bugs in conjugate gradient and `linear_solve`
  [PR #358](https://github.com/aai-institute/pyDVL/pull/358)
- Fix installation of dev requirements for Python3.10
  [PR #382](https://github.com/aai-institute/pyDVL/pull/382)
- Improvements to IF documentation
  [PR #371](https://github.com/aai-institute/pyDVL/pull/371)

## 0.6.1 - 🏗 Bug fixes and small improvements

- Fix parsing keyword arguments of `compute_semivalues` dispatch function
  [PR #333](https://github.com/aai-institute/pyDVL/pull/333)
- Create new `RayExecutor` class based on the concurrent.futures API,
  use the new class to fix an issue with Truncated Monte Carlo Shapley
  (TMCS) starting too many processes and dying, plus other small changes
  [PR #329](https://github.com/aai-institute/pyDVL/pull/329)
- Fix creation of GroupedDataset objects using the `from_arrays`
  and `from_sklearn` class methods 
  [PR #324](https://github.com/aai-institute/pyDVL/pull/334)
- Fix release job not triggering on CI when a new tag is pushed
  [PR #331](https://github.com/aai-institute/pyDVL/pull/331)
- Added alias `ApproShapley` from Castro et al. 2009 for permutation Shapley
  [PR #332](https://github.com/aai-institute/pyDVL/pull/332)

## 0.6.0 - 🆕 New algorithms, cleanup and bug fixes 🏗

- Fixes in `ValuationResult`: bugs around data names, semantics of
  `empty()`, new method `zeros()` and normalised random values
  [PR #327](https://github.com/aai-institute/pyDVL/pull/327)
- **New method**: Implements generalised semi-values for data valuation,
  including Data Banzhaf and Beta Shapley, with configurable sampling strategies
  [PR #319](https://github.com/aai-institute/pyDVL/pull/319)
- Adds kwargs parameter to `from_array` and `from_sklearn` Dataset and
  GroupedDataset class methods
  [PR #316](https://github.com/aai-institute/pyDVL/pull/316)
- PEP-561 conformance: added `py.typed`
  [PR #307](https://github.com/aai-institute/pyDVL/pull/307)
- Removed default non-negativity constraint on least core subsidy
  and added instead a `non_negative_subsidy` boolean flag.
  Renamed `options` to `solver_options` and pass it as dict.
  Change default least-core solver to SCS with 10000 max_iters.
  [PR #304](https://github.com/aai-institute/pyDVL/pull/304)
- Cleanup: removed unnecessary decorator `@unpackable`
  [PR #233](https://github.com/aai-institute/pyDVL/pull/233)
- Stopping criteria: fixed problem with `StandardError` and enable proper
  composition of index convergence statuses. Fixed a bug with `n_jobs` in
  `truncated_montecarlo_shapley`.
  [PR #300](https://github.com/aai-institute/pyDVL/pull/300) and
  [PR #305](https://github.com/aai-institute/pyDVL/pull/305)
- Shuffling code around to allow for simpler user imports, some cleanup and
  documentation fixes.
  [PR #284](https://github.com/aai-institute/pyDVL/pull/284)
- **Bug fix**: Warn instead of raising an error when `n_iterations`
  is less than the size of the dataset in Monte Carlo Least Core
  [PR #281](https://github.com/aai-institute/pyDVL/pull/281)

## 0.5.0 - 💥 Fixes, nicer interfaces and... more breaking changes 😒

- Fixed parallel and antithetic Owen sampling for Shapley values. Simplified
  and extended tests.
  [PR #267](https://github.com/aai-institute/pyDVL/pull/267)
- Added `Scorer` class for a cleaner interface. Fixed minor bugs around
  Group-Testing Shapley, added more tests and switched to cvxpy for the solver.
  [PR #264](https://github.com/aai-institute/pyDVL/pull/264)
- Generalised stopping criteria for valuation algorithms. Improved classes
  `ValuationResult` and `Status` with more operations. Some minor issues fixed.
  [PR #252](https://github.com/aai-institute/pyDVL/pull/250)
- Fixed a bug whereby `compute_shapley_values` would only spawn one process when
  using `n_jobs=-1` and Monte Carlo methods.
  [PR #270](https://github.com/aai-institute/pyDVL/pull/270)
- Bugfix in `RayParallelBackend`: wrong semantics for `kwargs`.
  [PR #268](https://github.com/aai-institute/pyDVL/pull/268)
- Splitting of problem preparation and solution in Least-Core computation.
  Umbrella function for LC methods.
  [PR #257](https://github.com/aai-institute/pyDVL/pull/257) 
- Operations on `ValuationResult` and `Status` and some cleanup
  [PR #248](https://github.com/aai-institute/pyDVL/pull/248)
- **Bug fix and minor improvements**: Fixes bug in TMCS with remote Ray cluster,
  raises an error for dummy sequential parallel backend with TMCS, clones model
  inside `Utility` before fitting by default, with flag `clone_before_fit` 
  to disable it, catches all warnings in `Utility` when `show_warnings` is 
  `False`. Adds Miner and Gloves toy games utilities
  [PR #247](https://github.com/aai-institute/pyDVL/pull/247)

## 0.4.0 - 🏭💥 New algorithms and more breaking changes

- GH action to mark issues as stale
  [PR #201](https://github.com/aai-institute/pyDVL/pull/201)
- Disabled caching of Utility values as well as repeated evaluations by default
  [PR #211](https://github.com/aai-institute/pyDVL/pull/211)
- Test and officially support Python version 3.9 and 3.10 
  [PR #208](https://github.com/aai-institute/pyDVL/pull/208)
- **Breaking change:** Introduces a class ValuationResult to gather and inspect
  results from all valuation algorithms
  [PR #214](https://github.com/aai-institute/pyDVL/pull/214)
- Fixes bug in Influence calculation with multidimensional input and adds new
  example notebook
  [PR #195](https://github.com/aai-institute/pyDVL/pull/195)
- **Breaking change**: Passes the input to `MapReduceJob` at initialization,
  removes `chunkify_inputs` argument from `MapReduceJob`, removes `n_runs`
  argument from `MapReduceJob`, calls the parallel backend's `put()` method for
  each generated chunk in `_chunkify()`, renames ParallelConfig's `num_workers`
  attribute to `n_local_workers`, fixes a bug in `MapReduceJob`'s chunkification
  when `n_runs` >= `n_jobs`, and defines a sequential parallel backend to run
  all jobs in the current thread
  [PR #232](https://github.com/aai-institute/pyDVL/pull/232)
- **New method**: Implements exact and monte carlo Least Core for data valuation,
  adds `from_arrays()` class method to the `Dataset` and `GroupedDataset`
  classes, adds `extra_values` argument to `ValuationResult`, adds
  `compute_removal_score()` and `compute_random_removal_score()` helper functions
  [PR #237](https://github.com/aai-institute/pyDVL/pull/237)
- **New method**: Group Testing Shapley for valuation, from _Jia et al. 2019_
  [PR #240](https://github.com/aai-institute/pyDVL/pull/240)
- Fixes bug in ray initialization in `RayParallelBackend` class
  [PR #239](https://github.com/aai-institute/pyDVL/pull/239)
- Implements "Egalitarian Least Core", adds [cvxpy](https://www.cvxpy.org/) as a
  dependency and uses it instead of scipy as optimizer
  [PR #243](https://github.com/aai-institute/pyDVL/pull/243)

## 0.3.0 - 💥 Breaking changes

- Simplified and fixed powerset sampling and testing
  [PR #181](https://github.com/aai-institute/pyDVL/pull/181)
- Simplified and fixed publishing to PyPI from CI
  [PR #183](https://github.com/aai-institute/pyDVL/pull/183)
- Fixed bug in release script and updated contributing docs.
  [PR #184](https://github.com/aai-institute/pyDVL/pull/184)
- Added Pull Request template
  [PR #185](https://github.com/aai-institute/pyDVL/pull/185)
- Modified Pull Request template to automatically link PR to issue
  [PR ##186](https://github.com/aai-institute/pyDVL/pull/186)
- First implementation of Owen Sampling, squashed scores, better testing
  [PR #194](https://github.com/aai-institute/pyDVL/pull/194)
- Improved documentation on caching, Shapley, caveats of values, bibtex
  [PR #194](https://github.com/aai-institute/pyDVL/pull/194)
- **Breaking change:** Rearranging of modules to accommodate for new methods
  [PR #194](https://github.com/aai-institute/pyDVL/pull/194)


## 0.2.0 - 📚 Better docs

Mostly API documentation and notebooks, plus some bugfixes.

### Added

In [PR #161](https://github.com/aai-institute/pyDVL/pull/161):
- Support for $$ math in sphinx docs.
- Usage of sphinx extension for external links (introducing new directives like
  `:gh:`, `:issue:` and `:tfl:` to construct standardised links to external
  resources).
- Only update auto-generated documentation files if there are changes. Some
  minor additions to `update_docs.py`.
- Parallelization of exact combinatorial Shapley.
- Integrated KNN shapley into the main interface `compute_shapley_values`.

### Changed

In [PR #161](https://github.com/aai-institute/pyDVL/pull/161):
- Improved main docs and Shapley notebooks. Added or fixed many docstrings,
  readme and documentation for contributors. Typos, grammar and style in code,
  documentation and notebooks.
- Internal renaming and rearranging in the parallelization and caching modules.

### Fixed

- Bug in random matrix generation
  [PR #161](https://github.com/aai-institute/pyDVL/pull/161).
- Bugs in MapReduceJob's `_chunkify` and `_backpressure` methods
  [PR #176](https://github.com/aai-institute/pyDVL/pull/176).


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


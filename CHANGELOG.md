# Changelog

## Unreleased


### Added

- Simple result serialization to resume computation of values
  [PR #666](https://github.com/aai-institute/pyDVL/pull/666)
- Simple memory monitor / reporting
  [PR #663](https://github.com/aai-institute/pyDVL/pull/663)
- New stopping criterion `MaxSamples`
  [PR #661](https://github.com/aai-institute/pyDVL/pull/661)
- Introduced `UtilityModel` and two implementations `IndicatorUtilityModel`
  and `DeepSetsUtilityModel` for data utility learning
  [PR #650](https://github.com/aai-institute/pyDVL/pull/650)
- Introduced the concept of `ResultUpdater` in order to allow samplers to
  declare the proper strategy to use by valuations 
  [PR #641](https://github.com/aai-institute/pyDVL/pull/641)
- Added Banzhaf precomputed values to some games.
  [PR #641](https://github.com/aai-institute/pyDVL/pull/641)
- Introduced new `IndexIterations`, for consistent usage across all
  `PowersetSamplers` [PR #641](https://github.com/aai-institute/pyDVL/pull/641)
- Added `run_removal_experiment` for easy removal experiments
  [PR #636](https://github.com/aai-institute/pyDVL/pull/636)
- Refactor Classwise Shapley valuation with the interfaces and sampler
  architecture [PR #616](https://github.com/aai-institute/pyDVL/pull/616)
- Refactor KNN Shapley values with the new interface
  [PR #610](https://github.com/aai-institute/pyDVL/pull/610)
  [PR #645](https://github.com/aai-institute/pyDVL/pull/645)
- Refactor MSR Banzhaf semivalues with the new sampler architecture.
  [PR #605](https://github.com/aai-institute/pyDVL/pull/605)
  [PR #641](https://github.com/aai-institute/pyDVL/pull/641)
- Refactor group-testing shapley values with new sampler architecture
  [PR #602](https://github.com/aai-institute/pyDVL/pull/602)
- Refactor least-core data valuation methods with more supported sampling
  methods and consistent interface.
  [PR #580](https://github.com/aai-institute/pyDVL/pull/580)
- Refactor Owen-Shapley valuation with new sampler architecture. Enable use of
  `OwenSamplers` with all semi-values
  [PR #597](https://github.com/aai-institute/pyDVL/pull/597)
  [PR #641](https://github.com/aai-institute/pyDVL/pull/641)
- New method `InverseHarmonicMeanInfluence`, implementation for the paper
  `DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and
    Diffusion Models`
  [PR #582](https://github.com/aai-institute/pyDVL/pull/582)
- Add new backend implementations for influence computation to account for
  block-diagonal approximations
  [PR #582](https://github.com/aai-institute/pyDVL/pull/582)
- Extend `DirectInfluence` with block-diagonal and Gauss-Newton approximation
  [PR #591](https://github.com/aai-institute/pyDVL/pull/591)
- Extend `LissaInfluence` with block-diagonal and Gauss-Newton approximation
  [PR #593](https://github.com/aai-institute/pyDVL/pull/593)
- Extend `NystroemSketchInfluence` with block-diagonal and Gauss-Newton
  approximation
  [PR #596](https://github.com/aai-institute/pyDVL/pull/596)
- Extend `ArnoldiInfluence` with block-diagonal and Gauss-Newton
  approximation
  [PR #598](https://github.com/aai-institute/pyDVL/pull/598)
- Extend `CgInfluence` with block-diagonal and Gauss-Newton approximation
  [PR #601](https://github.com/aai-institute/pyDVL/pull/601)

### Fixed

- Fixed `show_warnings=False` not being respected in subprocesses. Introduced
  `suppress_warninigs` decorator for more flexibility
  [PR #647](https://github.com/aai-institute/pyDVL/pull/647)
  [PR #662](https://github.com/aai-institute/pyDVL/pull/662)
- Fixed several bugs in diverse stopping criteria, including: iteration counts,
  computing completion, resetting, nested composition
  [PR #641](https://github.com/aai-institute/pyDVL/pull/641)
  [PR #650](https://github.com/aai-institute/pyDVL/pull/650)
- Fixed all weights of all samplers to ensure that mix-and-matching samplers and
  semi-value methods always works, for all possible combinations
  [PR #641](https://github.com/aai-institute/pyDVL/pull/641)
- Fixed a bug whereby progress bars would not report the last step and remain
  incomplete [PR #641](https://github.com/aai-institute/pyDVL/pull/641)
- Fixed the analysis of the adult dataset in the Data-OOB notebook
  [PR #636](https://github.com/aai-institute/pyDVL/pull/636)
- Replace `np.float_` with `np.float64` and `np.alltrue` with `np.all`,
  as the old aliases are removed in NumPy 2.0
  [PR #604](https://github.com/aai-institute/pyDVL/pull/604)
- Fix a bug in pydvl.utils.numeric.random_subset where 1 - q was used instead of q
  as the probability of an element being sampled
  [PR #597](https://github.com/aai-institute/pyDVL/pull/597)
- Fix a bug in the calculation of variance estimates for MSR Banzhaf
  [PR #605](https://github.com/aai-institute/pyDVL/pull/605)
- Fix a bug in KNN Shapley values. See [Issue 613](https://github.com/aai-institute/pyDVL/issues/613) for details.
- Backport the KNN Shapley fix to the `value` module
  [PR #633](https://github.com/aai-institute/pyDVL/pull/633) 

### Changed

- Slicing, comparing and setting of `ValuationResult` behave in a more 
  natural and consistent way
  [PR #660](https://github.com/aai-institute/pyDVL/pull/660) 
  [PR #666](https://github.com/aai-institute/pyDVL/pull/666)
- Switched all semi-value coefficients and sampler weights to log-space in
  order to avoid overflows
  [PR #643](https://github.com/aai-institute/pyDVL/pull/643)
- Updated and rewrote some of the MSR banzhaf notebook
  [PR #641](https://github.com/aai-institute/pyDVL/pull/641)
- Updated Least-Core notebook
  [PR #641](https://github.com/aai-institute/pyDVL/pull/641)
- Updated Shapley spotify notebook
  [PR #628](https://github.com/aai-institute/pyDVL/pull/628)
- Updated Data Utility notebook
  [PR #650](https://github.com/aai-institute/pyDVL/pull/650)
- Restructured and generalized `StratifiedSampler` to allow using heuristics,
  thus subsuming Variance-Reduced stratified sampling into a unified framework.
  Implemented the heuristics proposed in that paper
  [PR #641](https://github.com/aai-institute/pyDVL/pull/641)
- Uniformly distribute test points across processes for KNNShapley. Fail for
  `GroupedDataset` [PR #632](https://github.com/aai-institute/pyDVL/pull/632)
- Introduced the concept of logical vs data indices for `Dataset`, and
  `GroupedDataset`, fixing inconsistencies in how the latter operates on indices.
  Also, both now return objects of the same type when slicing.
  [PR #631](https://github.com/aai-institute/pyDVL/pull/631)
  [PR #648](https://github.com/aai-institute/pyDVL/pull/648)
- Use tighter bounds for the calculation of the minimal sample size that guarantees
  an epsilon-delta approximation in group testing (Jia et al. 2023)
  [PR #602](https://github.com/aai-institute/pyDVL/pull/602)
- Dropped black, isort and pylint from the CI pipeline, in favour of ruff
  [PR #633](https://github.com/aai-institute/pyDVL/pull/633)
- **Breaking Changes**
  - Changed `DataOOBValuation` to only accept bagged models
    [PR #636](https://github.com/aai-institute/pyDVL/pull/636)
  - Dropped support for python 3.8 after EOL
    [PR #633](https://github.com/aai-institute/pyDVL/pull/633)
  - Rename parameter `hessian_regularization` of `DirectInfluence`
    to `regularization` and change the type annotation to allow
    for block-wise regularization parameters
    [PR #591](https://github.com/aai-institute/pyDVL/pull/591)
  - Rename parameter `hessian_regularization` of `LissaInfluence`
    to `regularization` and change the type annotation to allow
    for block-wise regularization parameters
    [PR #593](https://github.com/aai-institute/pyDVL/pull/593)
  - Remove parameter `h0` from init of `LissaInfluence`
    [PR #593](https://github.com/aai-institute/pyDVL/pull/593)
  - Rename parameter `hessian_regularization` of `NystroemSketchInfluence`
    to `regularization` and change the type annotation to allow
    for block-wise regularization parameters
    [PR #596](https://github.com/aai-institute/pyDVL/pull/596)
  - Renaming of parameters of `ArnoldiInfluence`,
    `hessian_regularization` -> `regularization` (modify type annotation),
    `rank_estimate` -> `rank`
    [PR #598](https://github.com/aai-institute/pyDVL/pull/598)
  - Remove functions remove obsolete functions 
    `lanczos_low_rank_hessian_approximation`, `model_hessian_low_rank`
    from `influence.torch.functional`
    [PR #598](https://github.com/aai-institute/pyDVL/pull/598)
  - Renaming of parameters of `CgInfluence`,
    `hessian_regularization` -> `regularization` (modify type annotation),
    `pre_conditioner` -> `preconditioner`,
    `use_block_cg` -> `solve_simultaneously`
    [PR #601](https://github.com/aai-institute/pyDVL/pull/601)
  - Remove parameter `x0` from `CgInfluence`
    [PR #601](https://github.com/aai-institute/pyDVL/pull/601)
  - Rename module 
    `influence.torch.pre_conditioner` -> `influence.torch.preconditioner`
    [PR #601](https://github.com/aai-institute/pyDVL/pull/601)
  - Refactor preconditioner:
    - renaming `PreConditioner` -> `Preconditioner`
    - fit to `TensorOperator`
    [PR #601](https://github.com/aai-institute/pyDVL/pull/601)
  
  
## 0.9.2 - 🏗  Bug fixes, logging improvement

### Added

- Add progress bars to the computation of `LazyChunkSequence` and
  `NestedLazyChunkSequence`
  [PR #567](https://github.com/aai-institute/pyDVL/pull/567)
- Add a device fixture for `pytest`, which depending on the availability and
  user input (`pytest --with-cuda`) resolves to cuda device
  [PR #574](https://github.com/aai-institute/pyDVL/pull/574)

### Fixed

- Fixed logging issue in decorator `log_duration`
  [PR #567](https://github.com/aai-institute/pyDVL/pull/567)
- Fixed missing move of tensors to model device in `EkfacInfluence`
  implementation [PR #570](https://github.com/aai-institute/pyDVL/pull/570)
- Missing move to device of `preconditioner` in `CgInfluence` implementation
  [PR #572](https://github.com/aai-institute/pyDVL/pull/572)
- Raise a more specific error message, when a `RunTimeError` occurs in
  `torch.linalg.eigh`, so the user can check if it is related to a known
  issue
  [PR #578](https://github.com/aai-institute/pyDVL/pull/578)
- Fix an edge case (empty train data) in the test
  `test_classwise_scorer_accuracies_manual_derivation`, which resulted
  in undefined behavior (`np.nan` to `int` conversion with different results
  depending on OS)
  [PR #579](https://github.com/aai-institute/pyDVL/pull/579)

### Changed

- Changed logging behavior of iterative methods `LissaInfluence` and
  `CgInfluence` to warn on not achieving desired tolerance within `maxiter`,
  add parameter `warn_on_max_iteration` to set the level for this information
  to `logging.DEBUG`
  [PR #567](https://github.com/aai-institute/pyDVL/pull/567)

## 0.9.1 - Bug fixes, logging improvement

### Fixed

- `FutureWarning` for `ParallelConfig` constantly raised without actually
  instantiating the object
  [PR #562](https://github.com/aai-institute/pyDVL/pull/562)

## 0.9.0 - 🆕 New methods, better docs and bugfixes 📚🐞

### Added

- New method `MSR Banzhaf` with accompanying notebook, and new stopping
  criterion `RankCorrelation`
  [PR #520](https://github.com/aai-institute/pyDVL/pull/520)
- New method: `NystroemSketchInfluence`
  [PR #504](https://github.com/aai-institute/pyDVL/pull/504)
- New preconditioned block variant of conjugate gradient
  [PR #507](https://github.com/aai-institute/pyDVL/pull/507)
- Improvements to documentation: fixes, links, text, example gallery, LFS and
  more [PR #532](https://github.com/aai-institute/pyDVL/pull/532),
  [PR #543](https://github.com/aai-institute/pyDVL/pull/543)
- Glossary of data valuation and influence terms in the documentation
  [PR #537](https://github.com/aai-institute/pyDVL/pull/537
- Documentation about writing notes for new features, changes or deprecations
  [PR #557](https://github.com/aai-institute/pyDVL/pull/557)

### Fixed

- Bug in `LissaInfluence`, when not using CPU device
  [PR #495](https://github.com/aai-institute/pyDVL/pull/495)
- Memory issue with `CgInfluence` and `ArnoldiInfluence`
  [PR #498](https://github.com/aai-institute/pyDVL/pull/498)
- Raising specific error message with install instruction, when trying to load
  `pydvl.utils.cache.memcached` without `pymemcache` installed.
  If `pymemcache` is available, all symbols from `pydvl.utils.cache.memcached`
  are available through `pydvl.utils.cache`
  [PR #509](https://github.com/aai-institute/pyDVL/pull/509)

### Changed

- Add property `model_dtype` to instances of type `TorchInfluenceFunctionModel`
- Bump versions of CI actions to avoid warnings
  [PR #502](https://github.com/aai-institute/pyDVL/pull/502)
- Add Python Version 3.11 to supported versions
  [PR #510](https://github.com/aai-institute/pyDVL/pull/510)
- Documentation improvements and cleanup
  [PR #521](https://github.com/aai-institute/pyDVL/pull/521),
  [PR #522](https://github.com/aai-institute/pyDVL/pull/522)
- Simplified parallel backend configuration
  [PR #549](https://github.com/mkdocstrings/mkdocstrings/issues/615)

## 0.8.1 - 🆕 🏗  New method and notebook, Games with exact shapley values, bug fixes and cleanup

### Added

- Implement new method: `EkfacInfluence`
  [PR #451](https://github.com/aai-institute/pyDVL/issues/451)
- New notebook to showcase ekfac for LLMs
  [PR #483](https://github.com/aai-institute/pyDVL/pull/483)
- Implemented exact games in Castro et al. 2009 and 2017
  [PR #341](https://github.com/appliedAI-Initiative/pyDVL/pull/341)

### Fixed

- Bug in using `DaskInfluenceCalcualator` with `TorchnumpyConverter`
  for single dimensional arrays
  [PR #485](https://github.com/aai-institute/pyDVL/pull/485)
- Fix implementations of `to` methods of `TorchInfluenceFunctionModel`
  implementations [PR #487](https://github.com/aai-institute/pyDVL/pull/487)
- Fixed bug with checking for converged values in semivalues
  [PR #341](https://github.com/appliedAI-Initiative/pyDVL/pull/341)

### Changed

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
- Sequential batch-wise computation and write to disk with
  `SequentialInfluenceCalculator`
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

### Fixed

- Fix initialization of `data_names` in `ValuationResult.zeros()`
  [PR #443](https://github.com/aai-institute/pyDVL/pull/443)

### Changed

- No longer using docker within tests to start a memcached server
  [PR #444](https://github.com/aai-institute/pyDVL/pull/444)
- Using pytest-xdist for faster local tests
  [PR #440](https://github.com/aai-institute/pyDVL/pull/440)
- Improvements and fixes to notebooks
  [PR #436](https://github.com/aai-institute/pyDVL/pull/436)
- Refactoring of parallel module. Old imports will stop working in v0.9.0
  [PR #421](https://github.com/aai-institute/pyDVL/pull/421)

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


# Changelog

## Unreleased

### Added

- Support for $$ math in sphinx docs.
- Usage of sphinx extension for external links (introducing new directives like
  `:gh:`, `:issue:` and `:tfl:` to construct standardised links to external
  resources).
- Only update auto-generated documentation files if there are changes. Some
  minor additions to `update_docs.py`.
- Parallelization of exact combinatorial Shapley.
- Integrated KNN shapley into the main interface `compute_shapley_values`.

### Changed

- Improved main docs and Shapley notebooks. Added or fixed many docstrings,
  readme and documentation for contributors. Typos, grammar and style in code,
  documentation and notebooks.
- Internal renaming and rearranging in the parallelization and caching modules.

### Fixed

- Bug in random matrix generation


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

# Changelog

## Development

### Bug fixes

- Fixed bugs in MapReduceJob's `_chunkify` and `_backpressure` methods [PR #176](https://github.com/appliedAI-Initiative/pyDVL/pull/176)

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

# pyDVL

[![CI](https://github.com/appliedAI-Initiative/valuation/actions/workflows/tox.yaml/badge.svg)](https://github.com/appliedAI-Initiative/valuation/actions/workflows/tox.yaml) 


Welcome to the pyDVL library for data valuation!

Refer to our [documentation](https://appliedAI-Initiative.github.io/valuation) for more detailed information.

# Installation

To install the latest release use:

```shell
$ pip install pydvl
```

For more instructions and information refer to the [Installing pyDVL section](https://appliedAI-Initiative.github.io/valuation/install.html)
of the documentation.

# Usage

pyDVL requires Memcached in order to cache certain results and speed-up computation.

You need to run it either locally or using Docker:

```shell
docker container run -it --rm -p 11211:11211 memcached:latest -v
```

Caching is enabled by default but can be disabled if not needed or desired. 

For more instructions and information refer to the [Getting Started section](https://appliedAI-Initiative.github.io/valuation/getting-started.html) 
of the documentation 

Refer to the notebooks in the [notebooks](notebooks) folder for usage examples.

# Contributing

Please open new issues for bugs, feature requests and extensions. See more details about the structure and
workflow in the [developer's readme](README-dev.md).

# To do

* fix all 'em broken things.
* pytest plugin for algorithms with epsilon,delta guarantees:
  run n times, expect roughly n*delta failures at most.

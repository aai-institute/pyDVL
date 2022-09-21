<p align="center">
    <img alt="pyDVL" src="logo.svg" width="200"/>
</p>

<p align="center">
    A library for data valuation.
</p>

<p align="center">
    <a href="https://github.com/appliedAI-Initiative/valuation/actions/workflows/tox.yaml"><img src="https://github.com/appliedAI-Initiative/valuation/actions/workflows/tox.yaml/badge.svg" alt="Build Status" /></a>
</p>

<p align="center">
    <a href="https://appliedAI-Initiative.github.io/valuation">Docs</a>
</p>

# Installation

To install the latest release use:

```shell
$ pip install pyDVL
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

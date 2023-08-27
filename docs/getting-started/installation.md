---
title: Installing pyDVL
alias: 
    name: installation
    text: Installing pyDVL
---

# Installing pyDVL

To install the latest release use:

```shell
pip install pyDVL
```

To use all features of influence functions use instead:

```shell
pip install pyDVL[influence]
```

This includes a dependency on [PyTorch](https://pytorch.org/) (Version 2.0 and
above) and thus is left out by default.

In case that you have a supported version of CUDA installed (v11.2 to 11.8 as of
this writing), you can enable eigenvalue computations for low-rank approximations
with [CuPy](https://docs.cupy.dev/en/stable/index.html) on the GPU by using:

```shell
pip install pyDVL[cupy]
```

If you use a different version of CUDA, please install CuPy
[manually](https://docs.cupy.dev/en/stable/install.html).

In order to check the installation you can use:

```shell
python -c "import pydvl; print(pydvl.__version__)"
```

You can also install the latest development version from
[TestPyPI](https://test.pypi.org/project/pyDVL/):

```shell
pip install pyDVL --index-url https://test.pypi.org/simple/
```

## Dependencies

pyDVL requires Python >= 3.8, [Memcached](https://memcached.org/) for caching
and [Ray](https://ray.io) for parallelization in a cluster (locally it uses joblib).
Additionally, the [Influence functions][pydvl.influence] module requires PyTorch
(see [[installation]]).

ray is used to distribute workloads across nodes in a cluster (it can be used
locally as well, but for this we recommend joblib instead). Please follow the
instructions in their documentation to set up the cluster.

## Setting up the cache

[memcached](https://memcached.org/) is an in-memory key-value store accessible
over the network. pyDVL uses it to cache the computation of the utility function
and speed up some computations (in particular, semi-value computations with the
[PermutationSampler][pydvl.value.sampler.PermutationSampler] but other methods
may benefit as well).

You can either install it as a package or run it inside a docker container (the
simplest). For installation instructions, refer to the [Getting
started](https://github.com/memcached/memcached/wiki#getting-started) section in
memcached's wiki. Then you can run it with:

```shell
memcached -u user
```

To run memcached inside a container in daemon mode instead, do:

```shell
docker container run -d --rm -p 11211:11211 memcached:latest
```

!!! tip "Using the cache"
    Continue reading about the cache in the [First Steps](first-steps.md#caching)
    and the documentation for the [caching module][pydvl.utils.caching].

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

You can also install the latest development version from
[TestPyPI](https://test.pypi.org/project/pyDVL/):

```shell
pip install pyDVL --index-url https://test.pypi.org/simple/
```

In order to check the installation you can use:

```shell
python -c "import pydvl; print(pydvl.__version__)"
```

## Dependencies

pyDVL requires Python >= 3.8, [numpy](https://numpy.org/),
[scikit-learn](https://scikit-learn.org/stable/), [scipy](https://scipy.org/),
[cvxpy](https://www.cvxpy.org/) for the Core methods,
and [joblib](https://joblib.readthedocs.io/en/stable/)
for parallelization locally. Additionally,the [Influence functions][pydvl.influence]
module requires PyTorch (see [[installation#extras]]).

### Extras

pyDVL has a few [extra](https://peps.python.org/pep-0508/#extras) dependencies
that can be optionally installed:

- `influence`:

    To use all features of influence functions use instead:
    
    ```shell
    pip install pyDVL[influence]
    ```
    
    This includes a dependency on [PyTorch](https://pytorch.org/) (Version 2.0 and
    above) and thus is left out by default.

- `cupy`:

    In case that you have a supported version of CUDA installed (v11.2 to 11.8 as of
    this writing), you can enable eigenvalue computations for low-rank approximations
    with [CuPy](https://docs.cupy.dev/en/stable/index.html) on the GPU by using:
    
    ```shell
    pip install pyDVL[cupy]
    ```
  
    This installs [cupy-cuda11x](https://pypi.org/project/cupy-cuda11x/).
    
    If you use a different version of CUDA, please install CuPy
    [manually](https://docs.cupy.dev/en/stable/install.html).

- `ray`:

    If you want to use [Ray](https://www.ray.io/) to distribute data valuation
    workloads across nodes in a cluster (it can be used locally as well,
    but for this we recommend joblib instead) install pyDVL using:

    ```shell
    pip install pyDVL[ray]
    ```

    see [[first-steps#ray]] for more details on how to use it.

- `memcached`:

    If you want to use [Memcached](https://memcached.org/) for caching
    utility evaluations, use:
  
    ```shell
    pip install pyDVL[memcached]
    ```
    
    This installs [pymemcache](https://github.com/pinterest/pymemcache) additionally. 
    Be aware, that you still have to start a memcached server manually.

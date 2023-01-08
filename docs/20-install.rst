.. _pyDVL Installation:

================
Installing pyDVL
================

To install the latest release use:

.. code-block:: shell

    pip install pyDVL

To use all features of influence functions use instead:

.. code-block:: shell

    pip install pyDVL[influence]

This includes a dependency on `PyTorch <https://pytorch.org/>`_ and thus is left
out by default.

In order to check the installation you can use:

.. code-block:: shell

    python -c "import pydvl; print(pydvl.__version__)"

You can also install the latest development version from
`TestPyPI <https://test.pypi.org/project/pyDVL/>`_:

.. code-block:: shell

    pip install pyDVL --index-url https://test.pypi.org/simple/

Dependencies
============

pyDVL requires Python >= 3.8, `Memcached <https://memcached.org/>`_ for caching
and `ray <https://ray.io>`_ for parallelization. Additionally,
:mod:`Influence functions<pydvl.influence>` requires PyTorch (see
:ref:`pyDVL Installation`).

ray is used to distribute workloads both locally and across nodes. Please follow
the instructions in their documentation for installation.

.. _caching setup:

Setting up the cache
====================

memcached is an in-memory key-value store accessible over the network. pyDVL
uses it to cache certain results and speed-up the computations. You can either
install it as a package or run it inside a docker container (the simplest). For
installation instructions, refer to `Getting started
<https://github.com/memcached/memcached/wiki#getting-started>`_ in memcached's
wiki. Then you can run it with:

.. code-block:: shell

   memcached -u user

To run memcached inside a container in daemon mode instead, do:

.. code-block:: shell

    docker container run -d --rm -p 11211:11211 memcached:latest

.. warning::
   To read more about caching and how it might affect your usage, in particular
   about cache reuse and its pitfalls, please the documentation for the module
   :mod:`pydvl.utils.caching`.

What's next
===========

- Read on :ref:`data valuation`.
- Read on :ref:`influence functions <influence>`.
- Browse the :ref:`examples`.

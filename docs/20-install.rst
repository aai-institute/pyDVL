.. _pyDVL Installation:

================
Installing pyDVL
================

To install the latest release use:

.. code-block:: shell

    pip install pyDVL

You can also install the latest development version from `TestPyPI <https://test.pypi.org/project/pyDVL/>`_:

.. code-block:: shell

    pip install pyDVL --index-url https://test.pypi.org/simple/

To use all features of influence functions execute:

.. code-block:: shell

    pip install pyDVL[influence]

This includes a PyTorch and thus is left out by default.

In order to check the installation you can use:

.. code-block:: shell

    python -c "import valuation; print(pydvl.__version__)"

Dependencies
============

pyDVL requires Python >= 3.8, `Memcached <https://memcached.org/>`_ for caching
and `ray <https://ray.io>`_ for parallelization. Additionally, if you want to
use :mod:`Influence functions<pydvl.influence>` it also requires pytorch.

.. _caching setup:

Caching
=======

memcached is an in-memory key-value store accessible over the network. pyDVL
uses it to cache certain results and speed-up the computations. You can either
install it as a package or run it inside a docker container (the simplest).For
installation instructions, refer to "Getting started" in
`Memcached's wiki <https://github.com/memcached/memcached/wiki#getting-started>`_.
Then you can run it with:

.. code-block:: shell

   $ memcached -u user

To run memcached inside a container in daemon mode instead, do:

.. code-block:: shell

    $ docker container run -d --rm -p 11211:11211 memcached:latest

Caching is enabled by default for utility functions but can be disabled.

ray is used to distribute workloads both locally and across nodes. Please follow
the instructions in their documentation for installation.

What's next
===========

You should go to the :ref:`Getting Started <getting started>` section of the documentation.

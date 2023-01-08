.. _getting started:

===============
Getting started
===============

.. warning::
   Make sure you have read :ref:`the installation instructions
   <pyDVL Installation>` before using the library. In particular read about how
   caching and parallelization work, since they require additional setup.

pyDVL aims to be a repository of production-ready, reference implementations of
algorithms for data valuation and influence functions. You can read:

* :ref:`data valuation` for key objects and usage patterns for Shapley value
  computation and related methods.
* :ref:`influence` for instructions on how to compute influence functions (still
  in a pre-alpha state)

We only briefly introduce key concepts in the documentation. For a thorough
introduction and survey of the field, we refer to **the upcoming review** at the
:tfl:`TransferLab website <reviews/data-valuation>`.

Running the examples
====================

If you are somewhat familiar with the concepts of data valuation, you can start
by browsing our worked-out examples illustrating pyDVL's capabilities either:

- :ref:`In this documentation<examples>`.
- Using `binder <https://mybinder.org/>`_ notebooks, deployed from each
  example's page.
- Locally, by starting a jupyter server at the root of the project. You will
  have to install jupyter first manually since it's not a dependency of the
  library.

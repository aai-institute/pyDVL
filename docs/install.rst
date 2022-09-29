.. _pyDVL Installation:

================
Installing pyDVL
================

Dependencies
============

pyDVL requires the following dependencies:

- Python (>=3.8)
- Scikit-Learn
- Numpy
- Ray
- PyMemcached
- Tqdm
- Matplotlib

Optionally, if you want to use Influence functions it also requires:

- PyTorch

Installation
============

To install the latest release use:

.. code-block:: shell

    pip install pyDVL

To use all features of influence functions execute:

.. code-block:: shell

    pip install pyDVL[influence]

This includes a PyTorch and thus is left out by default.

In order to check the installation you can use:

.. code-block:: shell

    python -c "import valuation; print(valuation.__version__)"

What's next
===========

You should go to the :ref:`Getting Started <getting started>` section of the documentation.

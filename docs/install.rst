================
Installing pyDVL
================

Dependencies
============

pyDVL requires the following dependencies:

- Python (>=3.8)
- Scikit-Learn
- Numpy
- Joblib
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

If you want to use Influence Functions you should use:

.. code-block:: shell

    pip install pyDVL[influence]

In order to check the installation you can use:

.. code-block:: shell

    python -c "import valuation; print(valuation.__version__)"

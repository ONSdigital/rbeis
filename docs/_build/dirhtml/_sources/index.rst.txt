Rogers and Berriman Editing and Imputation System (RBEIS)
=========================================================

RBEIS is a method originally developed for imputing categorical data in relatively small social surveys with the intention of minimising conditional imputation variance. It is derived from CANCEIS, which is better suited to large datasets such as the Census.  This implementation of RBEIS works with `Pandas <https://pandas.pydata.org>`_ DataFrames.

RBEIS consists of a package ``rbeis``, containing classes used by all implementations of RBEIS, and subpackages ``rbeis.*``, containing implementations of the ``impute`` method using various backends.  Currently, only ``rbeis.pandas`` has a complete implementation, although future support is planned for PySpark in the ``rbeis.spark`` package.

To run an imputation, you will need to import ``impute`` and ``RBEISDistanceFunction`` by calling ``from rbeis.pandas import impute`` (using the Pandas backend) and ``from rbeis import RBEISDistanceFunction``.

Prerequisites
-------------

RBEIS was developed in an environment requiring support for Python 3.6.8, pandas 0.20.3, numpy 1.13.1 and wheel 0.29.0.  It may work with newer versions of these packages, but this is untested.

Installation
------------

The latest RBEIS wheel is available `via GitHub <https://github.com/y33les/rbeis/releases/latest>`_.  Download the ``.whl`` file and call ``pip install /path/to/wheel.whl`` to install it.

.. note:: RBEIS does not yet have ONS approval to be published through PyPI.  When this is given, it will be able to be installed easily using ``pip``.

RBEIS is licensed under the `MIT License <https://mit-license.org/>`_.

* :ref:`genindex`
* :ref:`search`

API Reference
=============

.. toctree::
   :maxdepth: 10

Key documents
=============

- `Method specification <spec/>`_
- `User notes <notes/>`_

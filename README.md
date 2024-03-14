# Rogers and Berriman Editing and Imputation System (RBEIS)

RBEIS implementations using Pandas and PySpark for SML

RBEIS is a method originally developed for imputing categorical data in relatively small social surveys with the intention of minimising conditional imputation variance. It is derived from CANCEIS, which is better suited to large datasets such as the Census.  This implementation of RBEIS works with `Pandas <https://pandas.pydata.org>`_ DataFrames.

## Prerequisites

- Python 3.6.8
- pandas 0.20.3
- numpy 1.13.1
- wheel 0.29.0

_(untested on newer versions; this is the environment in which RBEIS originally had to be developed)_

## Installation

~To install, simply call ``pip install rbeis``.~ _(awaiting approval to publish to PyPI)_

Whilst still in the pre-release stage, the latest RBEIS wheel can be manually downloaded [from GitHub](https://github.com/y33les/rbeis/releases/latest). Download the latest `*.whl` file, and run `pip install path/to/wheel`.

---

Our tests make use of Lemus and Stam's _Art History Textbook Data_ dataset, for which we are grateful to the authors for publishing and to [Tidy Tuesday](https://github.com/rfordatascience/tidytuesday/tree/master/data/2023/2023-01-17) for publicising.  We have included the original dataset at `tests/artists_original.csv`, a modified version at `tests/artists_unique_count.csv` and the same modified version with some data removed (for the purpose of testing imputation) at `tests/artists_unique_count_missing.csv`.

> Lemus S, Stam H (2022). arthistory: Art History Textbook Data. [https://github.com/saralemus7/arthistory](https://github.com/saralemus7/arthistory), [https://saralemus7.github.io/arthistory/](https://saralemus7.github.io/arthistory/).

"""
RBEIS is a method originally developed for imputing categorical data in relatively small social surveys with the intention of minimising conditional imputation variance. It is derived from CANCEIS, which is better suited to large datasets such as the Census.  This implementation of RBEIS works with [Pandas](https://pandas.pydata.org) DataFrames.
"""
__version__ = "0.1.0"

import warnings
from functools import partial

class RBEISInputException(Exception):
    """
    Exception raised in the case of faulty input to `impute` that is not covered by a TypeError.
    """
    pass


class RBEISInputWarning(UserWarning):
    """
    Warning raised in the case of redundant input to `impute`.
    """
    pass

class RBEISDistanceFunction:
    """
    Callable object encapsulating one of the six RBEIS distance functions, its threshold value (where appropriate), any pairs of values to be overridden (for DFs 4-6), and a weight by which to scale its result.
    
    RBEISDistanceFunctions have two public attributes: **f**, a two-argument function that computes the distance between two values, and **weight**, a numeric value by which the output of **f** is scaled (default value 1.0).

    - **df** (int): An integer from 1 to 6 corresponding to one of the standard RBEIS distance functions
    - **custom_map** (('a * 'a) * numeric dict): [Optional] A dictionary mapping pairs (2-tuples) of values to distances (value pair as key, distance as value), overriding the underlying standard distance function.  This is required for distance functions 4, 5 and 6.
    - **threshold** (numeric): [Optional] A threshold value for distance functions 2, 3, 5 and 6.
    - **weight** (numeric): [Optional] A weight value by which to scale the output of the distance function.

    .. note:: When using `custom_map`, please note that `(x,y)` &rarr; `z` does **not** imply that `(y,x)` &rarr; `z`.

    .. warning:: PLEASE NOTE that, when RBEIS calculates distances, the argument order is assumed to be (record, IGroup).  This is especially important when defining custom maps for DFs 4-6, which *do not* assume that `f(x,y) == f(y,x)`.

    _Example usage:_

    ```python
    myDF = RBEISDistanceFunction(6,
                                 custom_map = {(2, 3): 4, (8, 8): 0.25},
                                 threshold = 3,
                                 weight = 10)
    myDF(2, 2) # => 0
    myDF(2, 4) # => 5
    myDF(2, 8) # => 10
    myDF(2, 3) # => 40
    myDF(8, 8) # => 2.5
    ```
    """

    # Standard distance functions 1-3 (4-6 are 1-3 with custom overrides)
    # Where x is the IGroup's value, y is the current record's value, and m is a threshold value
    def _df1(self, x, y):
        """
        Distance Function 1: return 1 if `x` and `y` are not equal, otherwise 0.  This function (and its derivative, DF4) can be used with any data types that are comparable using `=`.

        - **x** ('a): A data point for comparison
        - **y** ('a): A data point for comparison

        _Example usages:_

        ```python
        _df1(2, 2) # => 0
        _df1(2, 4) # => 1
        ```
        """
        return int(x != y)

    def _df2(self, x, y, m=None):
        """
        Distance Function 2: return 1 if the difference between `x` and `y` is greater than the threshold `m`, otherwise 0.  This function (and its derivative, DF5) may only be used with numeric data.

        - **x** (numeric): A data point for comparison
        - **y** (numeric): A data point for comparison
        - **m** (numeric): A threshold value

        _Example usages:_

        ```python
        _df2(2, 2, 2) # => 0
        _df2(2, 3, 2) # => 0
        _df2(2, 8, 2) # => 1
        ```
        """
        if not (isinstance(x, Number) or isinstance(y, Number)):
            raise TypeError(
                "You have tried to call distance function 2, 3, 5 or 6, but your data is not numeric"
            )
        elif not (isinstance(m, Number)):
            raise TypeError(
                "You have provided a non-numeric threshold value to distance function 2, 3, 5 or 6"
            )
        else:
            return int(abs(x - y) > m)

    def _df3(self, x, y, m=None):
        """
        Distance Function 3: return 1 if the difference between `x` and `y` is greater than the threshold `m`, otherwise return the difference between `x` and `y`, divided by `m + 1`.  This function (and its derivative, DF6) may only be used with numeric data.

        - **x** (numeric): A data point for comparison
        - **y** (numeric): A data point for comparison
        - **m** (numeric): A threshold value

        _Example usages:_

        ```python
        _df3(2, 2, 3) # => 0
        _df3(2, 3, 3) # => 0.25
        _df3(2, 8, 3) # => 1
        ```
        """
        if not (isinstance(x, Number) or isinstance(y, Number)):
            raise TypeError(
                "You have tried to call distance function 2, 3, 5 or 6, but your data is not numeric"
            )
        elif not (isinstance(m, Number)):
            raise TypeError(
                "You have provided a non-numeric threshold value to distance function 2, 3, 5 or 6"
            )
        else:
            return 1.0 if abs(x - y) > m else abs(x - y) / (m + 1.0)

    f = _df1
    weight = 1.0

    def __init__(self, df, custom_map=None, threshold=None, weight=1.0):
        self.weight = float(weight)
        if df < 1 or df > 6:
            raise RBEISInputException(
                "The distance function must be an integer from 1 to 6 inclusive"
            )
        if df in [2, 3, 5, 6]:
            try:
                assert threshold
                if not (isinstance(threshold, Number)):
                    raise TypeError(
                        "Distance function thresholds must be a numeric type")
            except AssertionError:
                raise RBEISInputException(
                    "The chosen distance function requires a threshold value")
        elif threshold:
            warnings.warn(
                "You have supplied a threshold value for a distance function that does not require one",
                RBEISInputWarning,
            )
        if df >= 4:
            try:
                assert custom_map
                if not (all(
                        map(lambda x: isinstance(x, tuple),
                            custom_map.keys()))):
                    raise TypeError(
                        "Distance function overrides must be expressed in a dictionary whose keys are tuples representing pairs of values"
                    )
                if not (all(
                        map(lambda x: isinstance(x, Number),
                            custom_map.values()))):
                    raise TypeError(
                        "Distance function overrides must be expressed in a dictionary whose values are numeric"
                    )
            except AssertionError:
                raise RBEISInputException(
                    "You have chosen a distance funtion with overrides, but have not provided them"
                )
        elif custom_map:
            warnings.warn(
                "You have supplied custom mappings for a distance function that does not use them",
                RBEISInputWarning,
            )
        if not (isinstance(weight, Number)):
            raise TypeError("You have supplied a weight that is not numeric")
        if df == 1:
            self.f = self._df1
        elif df == 2:
            self.f = partial(self._df2, m=threshold)
        elif df == 3:
            self.f = partial(self._df3, m=threshold)
        elif df == 4:
            self.f = (lambda x, y: custom_map[(x, y)]
                      if (x, y) in custom_map.keys() else self._df1(x, y))
        elif df == 5:
            self.f = (
                lambda x, y: custom_map[(x, y)]
                if (x, y) in custom_map.keys() else self._df2(x, y, threshold))
        elif df == 6:
            self.f = (
                lambda x, y: float(custom_map[(x, y)])
                if (x, y) in custom_map.keys() else self._df3(x, y, threshold))

    def __call__(self, x, y):
        return self.weight * self.f(x, y)

# Example usage

> _**N.B.:** This guide assumes familiarity with the RBEIS algorithm.  If you are not familiar with how RBEIS works, please consult the methodological specification._

`impute` is the function that runs RBEIS on a Pandas DataFrame.  It takes four required positional arguments, and has several optional keyword arguments.  An example call to `impute` follows:

```python
from rbeis.pandas import impute, RBEISDistanceFunction
impute(pd.read_csv("my_data.csv"),
       "genre",
       ["jungle", "acid house", "UK garage"],
       {"artist": RBEISDistanceFunction(1, weight = 5),
        "bpm": RBEISDistanceFunction(5, custom_map = {(170, 180): 0, (140, 160): 100}, threshold = 5),
        "length": RBEISDistanceFunction(2, threshold = 1.25)},
       ratio = 1.5,
       in_place = False,
       keep_intermediates = True)
```

The first positional argument is a Pandas DataFrame, the dataset upon which to perform imputation.

The second is a string corresponding to the variable to be imputed.  This should match the name of one of the columns in the DataFrame.

The third is a list of possible values that the imputation variable can take.  In the above example, the imputation variable `"genre"` can take possible values from the list `["jungle", "acid house", "UK garage"]`.

The fourth positional argument represents the auxiliar variables that will be used in the imputation.  This is a dictionary whose keys are strings corresponding to the columns in the DataFrame to be used as auxiliary variables.  Its values are `RBEISDistanceFunction` objects representing the functions to be used to compare values within each auxiliary variable.  The `RBEISDistanceFunction` object is explained in greater depth below.

The three optional keyword arguments available to `impute` are `ratio`, `in_place` and `keep_intermediates`.

`ratio` is a numeric argument that allows RBEIS to accept donors with distances greater than the minimum.  If set, it allows RBEIS to accept values less than or equal to the value of `ratio` multiplied by the minimum distance.  It should be set to a number greater than 1, as values less than 1 would result in RBEIS looking for distances less than the minimum.  If a value less than 1 is given, an `RBEISInputException` will be thrown.  `ratio` has no default value, reverting to the original RBEIS behaviour of choosing only the minimum distance if it is not specified.

`in_place` is a Boolean argument (default `True`) which instructs `impute` to overwrite the original DataFrame with the results of the imputation.  If `in_place` is `False`, `impute` returns a new DataFrame with the imputed data included.

`keep_intermediates` is a Boolean argument (default `False`) which instructs `impute` not to delete the temporary columns it creates in the process of imputation.  Its primary use is for debugging.

## Distance functions

Distance functions are encapsulated in `RBEISDistanceFunction` objects, which specify which of the six RBEIS distance functions to use and contain any required threshold values, custom mappings and weights.  They can be constructed in several ways:

1. Standard distance function: `RBEISDistanceFunction(1)`
    - This creates a distance function exactly as in the RBEIS specification.  In this case, this is the RBEIS distance function 1.
1. Weighted distance function: `RBEISDistanceFunction(1, weight = 2)`
    - This creates a distance function with its output scaled by a given factor.  In this case, this is the RBEIS distance function 1 with its output scaled by a factor of 2.
1. Distance function with threshold: `RBEISDistanceFunction(3, threshold = 5)`
    - This creates one of the RBEIS distance functions (2, 3, 5 or 6) that requires a threshold value.  In this case, we create RBEIS distance function 3 with a threshold value of 5.
1. Distance function with overrides: `RBEISDistanceFunction(4, custom_map = {(1, 2): 10, (2, 1): 200})`
    - This creates one of the RBEIS distance functions (4, 5, or 6) that overrides certain pairs of values.  These overrides are represented as a dictionary whose keys are the pair of values to be overridden and whose values are the distances to return when those pairs of values are passed to the `RBEISDistanceFunction`.  Note that argument order matters here; in the example above the function as applied to `(1, 2)` returns a distance of 10, whereas the function applied to the `(2, 1)` returns a distance of 200.

The keyword arguments `weight`, `threshold` and `custom_map` can be used in combination with each other when necessary.  In the following example, RBEIS distance function 6 is constructed with its requisite threshold and overrides.  It is also given a `weight` of 4.

```python
RBEISDistanceFunction(6, custom_map = {(1, 2): 10, (5, 3): 400}, threshold = 2.5, weight = 4)
```

It is possible to provide redundant arguments, for example providing a threshold value to distance function 1.  These will not be used, and a warning will be displayed upon creation.

`RBEISDistanceFunction`s are callable, meaning that they can be used as any other function:

```python
myDF = RBEISDistanceFunction(6, custom_map = {(1, 2): 10, (5, 3): 400}, threshold = 2.5, weight = 4)
myDF(2,3)
```

# Example usage

`impute` is the function that runs RBEIS on a Pandas DataFrame.  It takes four required positional arguments, and has several optional keyword arguments.

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

It is possible to provide redundant arguments, for example providing a threshold value to distance function 1.  These will not be used, and a warning will be generated.

`RBEISDistanceFunction`s are callable, meaning that they can be used as any other function:

```python
myDF = RBEISDistanceFunction(6, custom_map = {(1, 2): 10, (5, 3): 400}, threshold = 2.5, weight = 4)
myDF(2,3)
```

## Example usage with HFS-/WAS-style ImpSpec

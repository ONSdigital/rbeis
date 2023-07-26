# Rogers and Berriman Editing and Imputation System (RBEIS)

## 1 Meta

- **Support Area:** Editing and Imputation
- **Method Theme:** Editing and Imputation
- **Last reviewed:** 2023-07-25

## 2 Terminology

<!--
A bulleted list of technical terms specific to this method that are used in the specification
-->

TODO

## 3 Summary

<!--
A brief (~100 words) summary of what your method does, how it achieves it, and what kinds of data it expects to receive and produce.  These will each be explored in greater depth in later sections, so make sure not to go into too much detail here
-->

RBEIS is a method originally developed for imputing categorical data in relatively small social surveys with the intention of minimising conditional imputation variance.  It is derived from CANCEIS, which is better suited to large datasets such as the Census.  RBEIS differs from CANCEIS by constructing donor pools for each IGroup, rather than for each record.  These donor pools are then further processed to be the same size as the IGroup.  This implementation of RBEIS expects a Pandas DataFrame as input.

## 4 Assumptions

<!--
A bulleted list of assumptions that your method makes about its inputs
-->

TODO
- DFs expect (record, IGroup) ordering
- DFs assume that categorical data is ordered, i.e. cat 1 is less than cat 2

## 5 Method input and output

<!--
A section detailing the kinds of data that your method expects to receive as input and produce as output, along with any related information (e.g. “fields must not contain null values”)
-->

### 5.1 Input records

<!--
Details about the expected input, including (if applicable) the expected fields within each record and the formats in which they are expected
-->

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

### 5.2 Output records

<!--
Details about the require output, including (if applicable) the expected fields within each record and the formats in which they are required
-->

TODO: Uncomment

<!--`impute` modifies its input DataFrame by adding a new column containing the imputed values for a given variable, named <code><em>&lt;variable&gt;</em>_imputed</code>.  If `in_place` is set to `False`, a new DataFrame containing this column is returned.-->

### 5.3 Error handling

<!--
Details about what the method should do in the event of various classes of errors
-->

RBEIS performs extensive checks on its inputs.  Where checks find incompatible or nonsensical inputs, an exception is raised and `impute` is terminated.  `TypeError`s are raised when inputs of inappropriate type are supplied. `RBEISInputException`s are raised where inputs do not make sense for RBEIS (e.g. choosing distance function 7 when RBEIS only provides six) or where required data is missing.  Where checks find unnecessary inputs (for example redundant optional arguments), a warning of class `RBEISInputWarning` is raised.

### 5.4 Metadata

<!--
Details about what other metadata should be provided by the method, e.g. the number of times a donor was used in imputation
-->

Currently, RBEIS collects no metadata by default.  If `keep_intermediates` is set to `True`, the temporary values calculated by the `impute` function are able to be inspected post-imputation.  These values are IGroup assignments, calculated distances and records to which each record may be a donor.

## 6 Method

<!--
A detailed, formal, prose description of your method including, where appropriate, the underlying mathematics.  This section is best broken up into multiple subsections, especially for more complex methods
-->

1. Assign each record requiring imputation to an imputation group ("IGroup") based on the values of its auxiliary variables.
1. For each record requiring imputation, calculate the distance between the values of its auxiliary variables and those of each IGroup.  Each auxiliary variable should have a distance function assigned to it, and may also be assigned a weight.  The total distance is calculated thus:

$$
\sum_{v \in A} w_{v}{f_{v}(v_{record},v_{IGroup})}
$$

  Where $A$ is the set of auxiliary variables, $w_v$ is the weight assigned to the auxiliary variable $v$, $f_v$ is the distance function assigned to the auxiliary variable $v$, $v_{record}$ is the value of the auxiliary variable for the current record, and $v_{IGroup}$ is the value of the auxiliary variable for the current IGroup.
1. Determine donors
1. Impute

- IGroup
- Auxiliary variable
- Distance
- Donor pool
- Donor
- Impute
- Distance function

## 7 Further information

<!--
If appropriate, a bulleted list of links to external documents that provide further information about the method
-->

- [Presentation to UNECE, September 2020](https://web.archive.org/web/20230725125247/https://unece.org/fileadmin/DAM/stats/documents/ece/ces/ge.58/2020/mtg1/SDE2020_T1-A_UK_Leather_Presentation.pdf)

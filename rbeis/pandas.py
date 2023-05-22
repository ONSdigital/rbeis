import numpy as np
import pandas as pd
import warnings
from functools import partial
from fractions import Fraction
from numpy.random import choice
from random import shuffle
from numbers import Number
from copy import deepcopy


class RBEISInputException(Exception):
    pass


class RBEISInputWarning(UserWarning):
    pass


class RBEISDistanceFunction:
    """
    RBEISDistanceFunction

    Callable object encapsulating one of the six RBEIS distance functions, its
    threshold value (where appropriate), any pairs of values to be overridden
    (for DFs 4-6), and a weight by which to scale its result.

    PLEASE NOTE that, in rbeis._calc_distances, the argument order is assumed to
    be (record, IGroup).  This is especially important when defining custom maps
    for DFs 4-6, which *do not* assume that f(x,y) == f(y,x).
    """

    # Standard distance functions 1-3 (4-6 are 1-3 with custom overrides)
    # Where x is the IGroup's value, y is the current record's value, and m is a threshold value
    def _df1(self, x, y):
        """
        _df1(x, y)

        x ('a): A data point for comparison
        y ('a): A data point for comparison

        Distance Function 1: return 1 if x and y are not equal, otherwise 0.
        This function (and its derivative, DF4) can be used with any data types
        that are comparable using =.

        e.g. _df1(2, 2) => 0
             _df1(2, 4) => 1
        """
        return int(x != y)

    def _df2(self, x, y, m=None):
        """
        _df2(x, y, m)

        x (numeric): A data point for comparison
        y (numeric): A data point for comparison
        m (numeric): A threshold value

        Distance Function 2: return 1 if the difference between x and y is
        greater than the threshold m, otherwise 0.  This function (and its
        derivative, DF5) may only be used with numeric data.

        e.g. _df2(2, 2, 2) => 0
             _df2(2, 3, 2) => 0
             _df2(2, 8, 2) => 1
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
        _df3(x, y, m)

        x (numeric): A data point for comparison
        y (numeric): A data point for comparison
        m (numeric): A threshold value

        Distance Function 3: return 1 if the difference between x and y is
        greater than the threshold m, otherwise return the difference between x
        and y, divided by m + 1.  This function (and its derivative, DF6) may
        only be used with numeric data.

        e.g. _df3(2, 2, 3) => 0
             _df3(2, 3, 3) => 0.25
             _df3(2, 8, 3) => 1
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
        """
        RBEISDistanceFunction(df, custom_map=None, threshold=None, weight=1.0)

                                     df (int): An integer from 1 to 6
                                               corresponding to one of the
                                               standard RBEIS distance functions
        custom_map (('a * 'a) * numeric dict): [Optional] A dictionary mapping
                                               pairs (2-tuples) of values to
                                               distances (value pair as key,
                                               distance as value), overriding
                                               the underlying standard distance
                                               function.  Note that (x,y) -> z
                                               does not imply that (y,x) -> z.
                                               This is required for distance
                                               functions 4, 5 and 6.
                          threshold (numeric): [Optional] A threshold value for
                                               distance functions 2, 3, 5 and 6.
                             weight (numeric): [Optional] A weight value by
                                               which to scale the output of the
                                               distance function

        Initialise a new RBEISDistanceFunction object with the above parameters.

        e.g. myDF = RBEISDistanceFunction(6,
                                          custom_map = {(2, 3): 4, (8, 8): 0.25},
                                          threshold = 3,
                                          weight = 10)
             myDF(2, 2) => 0
             myDF(2, 4) => 5
             myDF(2, 8) => 10
             myDF(2, 3) => 40
             myDF(8, 8) => 2.5
        """
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


def _check_missing_auxvars(data, aux_vars):
    """
    _check_missing_auxvars(data, aux_vars)


                                  data (pd.DataFrame): The dataset undergoing
                                                       imputation
          aux_vars (str * RBEISDistanceFunction dict): A dictionary whose keys
                                                       are strings corresponding
                                                       to the names of auxiliary
                                                       variables and whose values
                                                       are the
                                                       RBEISDistanceFunctions to
                                                       be used to compare
                                                       instances of each
                                                       auxiliary variable.

    Raise an RBEISInputException if the DataFrame contains any records for which
    any of the chosen auxiliary variables are missing.

    e.g. _check_missing_auxvars(pd.read_csv("my_data.csv"),
                                {"height": RBEISDistanceFunction(1,
                                                                 weight = 5),
                                 "length": RBEISDistanceFunction(5,
                                                                 custom_map = {(2, 3): 0,
                                                                               (8, 8): 0.25},
                                                                 threshold = 2)})
    """
    try:
        assert not (any([
            any(
                map(
                    lambda i: isinstance(i, float) and np.isnan(i),
                    data[k].tolist(),
                )) for k in aux_vars.keys()
        ]))
    except AssertionError:
        raise RBEISInputException(
            "Your dataset includes records for which the given auxiliary variables are missing"
        )


def _add_impute_col(data, imp_var):
    """
    _add_impute_col(data, imp_var)

    data (pd.DataFrame): The dataset undergoing imputation
          imp_var (str): The name of the variable to be imputed

    Prepare the DataFrame by adding a boolean column '_impute' indicating
    whether or not a record is to be imputed.  A record will be marked for
    imputation if a value for the specified imputation variable is missing.
    This function modifies the DataFrame in place, rather than returning a
    new DataFrame.

    e.g. _add_impute_col(pd.read_csv("my_data.csv"), "length")
    """
    # Note that data["_impute"] = np.isnan(data[imp_var]) does not work when imp_var is non-numeric
    data["_impute"] = data.apply(
        lambda r: np.isnan(r[imp_var])
        if isinstance(r[imp_var], Number) else False,
        axis=1,
    )


def _assign_igroups(data, aux_var_names):
    """
    _assign_igroups(data, aux_var_names)

         data (pd.DataFrame): The dataset undergoing imputation
    aux_var_names (str list): The names of the chosen auxiliary variables

    Add a column 'IGroup' containing integers representing the IGroup that each
    recipient record is assigned to.  This function modifies the DataFrame in
    place, rather than returning a new DataFrame.
    1. Add a column '_conc_aux' containing the string representation of the list
       of values corresponding to the record's auxiliary variables
    2. Group by '_conc_aux' and extract group integers from the internal grouper
       object, creating a new column '_IGroup' containing these integers
    3. Subtract 1 from each IGroup value, giving -1 for all non-recipient
       records and zero-indexing all others
    4. Remove column '_conc_aux'

    e.g. _assign_igroups(myDataFrame, ["height", "width", "length"]
    """
    data["_conc_aux"] = data.apply(
        lambda r: str(list(map(lambda v: r[v], aux_var_names)))
        if r["_impute"] else "",
        axis=1,
    )
    data["_IGroup"] = data.groupby("_conc_aux").grouper.group_info[0]
    data["_IGroup"] = data.apply(lambda r: r["_IGroup"] - 1, axis=1)
    del data["_conc_aux"]


def _get_igroup_aux_var(data, aux_var_name):
    """
    _get_igroup_aux_var(data, aux_var_name)

    data (pd.DataFrame): The dataset undergoing imputation
     aux_var_name (str): The name of the desired auxiliary variable

    Return a list containing each IGroup's value of a given auxiliary variable.
    The value at index i corresponds to the value for IGroup i.

    e.g. _get_igroup_aux_var(myDataFrame, "height") => [188, 154, 192, ...]
    """
    out = []
    for i in range(1 + data["_IGroup"].max()):
        out += (data[[
            "_IGroup", aux_var_name
        ]].drop_duplicates().query("_IGroup == " +
                                   str(i))[aux_var_name].values.tolist())
    return out


def _calc_distances(data, aux_vars):
    """
    _calc_distances(data, aux_vars)

                            data (pd.DataFrame): The dataset undergoing
                                                 imputation
    aux_vars (str * RBEISDistanceFunction dict): A dictionary whose keys are
                                                 strings corresponding to the
                                                 names of auxiliary variables
                                                 and whose values are the
                                                 RBEISDistanceFunctions to be
                                                 used to compare instances of
                                                 each auxiliary variable.

    Add a column '_distances' containing lists of calculated distances of each
    record's auxiliary variables from those of its IGroup.  This function
    modifies the DataFrame in place, rather than returning a new DataFrame.
    1. Create a list of dictionaries containing the values of each IGroup's
       auxiliary variables
    2. Add a column '_dists_temp' containing, for each potential donor record, a
       list of dictionaries of calculated distances for each auxiliary variable
    3. Add a column '_distances' containing, for each potential donor record, a
       list of calculated distances from its IGroup's auxiliary variables,
       taking into account the specified weights
    4. Remove column '_dists_temp'

    PLEASE NOTE that the argument order is assumed to be (record, IGroup).  This
    is especially important when using custom maps with distance functions 4-6,
    which *do not* assume that f(x,y) == f(y,x).

    e.g. _calc_distances(myDataFrame, {"length": RBEISDistanceFunction(2, threshold=1.5),
                                       "genre": RBEISDistanceFunction(1, weight=8),
                                       "rating": RBEISDistanceFunction(4, custom_map={(3,4): 0,
                                                                                      (1,1): 10}})
    """
    igroup_aux_vars = []
    vars_vals = list(
        zip(
            aux_vars.keys(),
            map(lambda k: _get_igroup_aux_var(data, k), aux_vars.keys()),
        ))
    for i in range(1 + data["_IGroup"].max()):
        igroup_aux_vars.append({k: v[i] for k, v in vars_vals})

    data["_dists_temp"] = data.apply(
        lambda r: list(
            map(
                lambda g: {
                    k: aux_vars[k](r[k], igroup_aux_vars[g][k])
                    for k in aux_vars.keys()
                },
                range(1 + data["_IGroup"].max()),
            )) if not (r["_impute"]) else [],
        axis=1,
    )
    data["_distances"] = data.apply(
        lambda r: list(map(lambda d: sum(d.values()), r["_dists_temp"]))
        if not (r["_impute"]) else [],
        axis=1,
    )
    del data["_dists_temp"]


def _calc_donors(data, ratio=None):
    """
    _calc_donors(data, ratio=None)

    data (pd.DataFrame): The dataset undergoing imputation
        ratio (numeric): [Optional] Instead of choosing the minimum distance,
                         choose records less than or equal to ratio * the
                         minimum distance.

    Add a column 'donor' containing a list of IGroup numbers to which each
    record is a donor.  This function modifies the DataFrame in place, rather
    than returning a new DataFrame.
    1. Calculate the distances less than or equal to which a record may be
       accepted as a donor to each respective IGroup.
    2. Zip each record's distances to the list of distances calculated in step
       1, and identify where the record distances are less than or equal to the
       maximum required IGroup distances, giving a list of indices where this is
       the case.

    e.g. _calc_donors(myDataFrame)
         _calc_donors(myDataFrame, ratio=2.5)
    """
    igroups_dists = np.array(
        data.query("not(_impute)")["_distances"].values.tolist()).T.tolist()
    max_donor_dists = list(
        map(
            lambda l: max([i for i in l if i <= ratio * min(l)])
            if ratio else min(l),
            igroups_dists,
        ))
    data["_donor"] = data.apply(
        lambda r: np.where(
            list(
                map(lambda p: p[0] <= p[1],
                    zip(r["_distances"], max_donor_dists))))[0].tolist()
        if not (r["_impute"]) else [],
        axis=1,
    )  # TODO: What if we have multiple minima?  Do we just choose the first, or choose one randomly?


def _get_donors(data, igroup):
    """
    _get_donors(data, igroup)

    data (pd.DataFrame): The dataset undergoing imputation
           igroup (int): The IGroup whose donors are required

    Return a list of indices corresponding to records in data that are donors to
    the specified IGroup.

    e.g. _get_donors(myDataFrame, 24) => [10, 14, 29, ...]
    """
    return list(map(lambda x: igroup in x, data["_donor"].values.tolist()))


def _get_freq_dist(data, imp_var, possible_vals, igroup):
    """
    _get_freq_dist(data, imp_var, possible_vals, igroup)

        data (pd.DataFrame): The dataset undergoing imputation
              imp_var (str): The name of the variable to be imputed
    possible_vals ('a list): A list of all possible values that imp_var can take
               igroup (int): The IGroup whose donors are required

    For a given IGroup, return a frequency distribution for each possible value
    of the variable to be imputed.  This takes the form of a list of the
    proportions of a given IGroup taken up by each possible value.

    e.g. _get_freq_fist(myDataFrame, "genre", ["jungle", "acid house", "UK garage"], 48) => [0.5, 0.3, 0.2]
    """
    pool = data[_get_donors(data, igroup)][imp_var].values.tolist()
    return list(
        map(
            lambda n: 0 if n == 0 else Fraction(n, len(pool)),
            map(
                lambda v: len([i for i in pool if i == v]),
                possible_vals,
            ),
        ))


def _freq_to_exp(data, freq_dist, igroup):
    """
    _freq_to_exp(data, freq_dist, igroup)

          data (pd.DataFrame): The dataset undergoing imputation
    freq_dist (Fraction list): The frequency distribution to convert
                 igroup (int): The IGroup corresponding to freq_dist

    Convert a frequency distribution to the expected numbers of occurrences for
    a given IGroup.

    e.g. _freq_to_exp(myDataFrame, [0.5, 0.3, 0.2], 48) => [100, 60, 40]
    """
    igroup_size = len(data.query("_IGroup==" + str(igroup)).values.tolist())
    return list(map(lambda f: f * igroup_size, freq_dist))


def _impute_igroup(data, exp_dist, possible_vals, igroup):
    """
    _impute_igroup(data, exp_dist, possible_vals, igroup)

         data (pd.DataFrame): The dataset undergoing imputation
    exp_dist (Fraction list): The expected values derived from the frequency
                              distribution using _freq_to_exp
     possible_vals ('a list): A list of all possible values that imp_var can take
                igroup (int): The IGroup whose values are to be imputed

    Return a set of imputed values for the given IGroup:
    1. For each of the possible values that the variable to be imputed can take,
       insert a number of values equal to the integer part of the expected value
       and subtract the integer part from the expected value
    2. Convert the fractional parts of the expected values back into
       probabilities
    3. Using these probabilities, draw a value at random, remove that value from
       the set of possible values to be imputed, and adjust the remaining
       probabilities accordingly
    4. Repeat step 3 until there are no more values still to impute
    5. Randomise the order of the list of imputed values, then return it

    e.g. _impute_igroup(myDataFrame, [100, 60, 40], ["jungle", "acid house", "UK garage"], 48)
    """
    out = []
    if all(map(lambda e: e == 0, exp_dist)):
        return out
    igroup_size = len(data.query("_IGroup==" + str(igroup)).values.tolist())
    for i in range(len(exp_dist)):
        if exp_dist[i] >= 1:
            out += [possible_vals[i]] * int(exp_dist[i])
            exp_dist[i] -= int(exp_dist[i])
    remaining = igroup_size - len(out)
    if remaining != 0:
        exp_dist = list(map(lambda e: Fraction(e, sum(exp_dist)), exp_dist))
        assert sum(exp_dist) == 1
        possible_vals = [
            possible_vals[i] for i in range(len(exp_dist)) if exp_dist[i] != 0
        ]
        exp_dist = [e for e in exp_dist if e != 0]
        for i in range(remaining):
            selected_val = choice(possible_vals, p=exp_dist)
            del exp_dist[possible_vals.index(selected_val)]
            exp_dist = list(map(lambda e: Fraction(e, sum(exp_dist)),
                                exp_dist))
            assert sum(exp_dist) == 1
            possible_vals = [v for v in possible_vals if v != selected_val]
            out.append(selected_val)
        # TODO: check that number of imputed values for each possible value is
        #       either int(exp) or int(exp)+1
    shuffle(out)
    return out


def impute(
    data,
    imp_var,
    possible_vals,
    aux_vars,
    ratio=None,
    in_place=True,
    keep_intermediates=False,
):
    """
    impute(data, imp_var, possible_vals, aux_vars, ratio=None, in_place=True,
           keep_intermediates=False)

                            data (pd.DataFrame): The dataset undergoing
                                                 imputation
                                  imp_var (str): The name of the variable to be
                                                 imputed
                        possible_vals ('a list): A list of all possible values
                                                 that imp_var can take
    aux_vars (str * RBEISDistanceFunction dict): A dictionary whose keys are
                                                 strings corresponding to the
                                                 names of auxiliary variables
                                                 and whose values are the
                                                 RBEISDistanceFunctions to be
                                                 used to compare instances of
                                                 each auxiliary variable.
                                ratio (numeric): [Optional] Instead of choosing
                                                 the minimum distance, choose
                                                 records less than or equal to
                                                 ratio * the minimum distance.
                                in_place (bool): [Optional, default True] If
                                                 True, modify the original
                                                 DataFrame in place.  If False,
                                                 return a new (deep) copy of the
                                                 DataFrame having undergone
                                                 imputation.
                      keep_intermediates (bool): [Optional, default False] If
                                                 True, retain the intermediate
                                                 columns created by this
                                                 implementation of RBEIS in the
                                                 process of imputation.  If
                                                 False, remove them from the
                                                 output.

    Impute missing values for a given variable using the Rogers & Berriman
    Editing and Imputation System (RBEIS).  By default, this function modifies
    the existing DataFrame in place, rather than returning a new DataFrame,
    unless in_place is set to False.  A high-level overview of the approach is
    given here (for more detail, see the documentation for each of the
    intermediate functions in this library):
    1. Identify values to be imputed
    2. Assign imputation groups ("IGroups") based on a given set of auxiliary
       variables
    3. Calculate how similar the auxiliary variables of each IGroup are to those
       of the potential donor records
    4. Assign the most similar records to the donor pools of the corresponding
       IGroups
    5. Impute values for each IGroup
    6. Insert imputed values into the original DataFrame

    e.g. impute(pd.read_csv("my_data.csv"),
                "genre",
                ["jungle", "acid house", "UK garage"],
                {"artist": RBEISDistanceFunction(1, weight = 5),
                 "bpm": RBEISDistanceFunction(5, custom_map = {(170, 180): 0, (140, 160): 100}, threshold = 5),
                 "length": RBEISDistanceFunction(2, threshold = 1.25)},
                ratio = 1.5,
                in_place = False,
                keep_intermediates = True)
    """

    # Input checks
    if not (isinstance(data, pd.DataFrame)):
        raise TypeError("Dataset must be a Pandas DataFrame")
    if not (isinstance(imp_var, str)):
        raise TypeError("Imputation variable name must be a string")
    if not (isinstance(possible_vals, list)):
        raise TypeError("Possible values must be contained in a list")
    if not (all(list(map(lambda v: v in possible_vals,data[imp_var].unique()))+[np.nan])):
        raise RBEISInputException("The column to undergo imputation contains values that are not included in possible_vals")
    if not (isinstance(aux_vars, dict)):
        raise TypeError(
            "aux_vars must be a dictionary whose keys are strings representing auxiliary vvariables and whose values are RBEISDistanceFunctions"
        )
    if not (all(map(lambda x: isinstance(x, str), aux_vars.keys()))):
        raise TypeError(
            "aux_vars must be a dictionary whose keys are strings containing auxiliary variable names"
        )
    if not (all(
            map(lambda x: isinstance(x, RBEISDistanceFunction),
                aux_vars.values()))):
        raise TypeError(
            "aux_vars must be a dictionary whose values are RBEISDistanceFunctions"
        )
    try:
        assert ratio
        if not (isinstance(ratio, Number)):
            raise TypeError("The ratio must be numeric")
        if ratio < 1:
            raise RBEISInputException("The ratio must be greater than 1")
    except AssertionError:
        pass
    try:
        assert in_place
        if not (isinstance(in_place, bool)):
            raise TypeError("in_place must be either True or False")
    except AssertionError:
        pass
    try:
        assert keep_intermediates
        if not (isinstance(keep_intermediates, bool)):
            raise TypeError("keep_intermediates must be either True or False")
    except AssertionError:
        pass

    # Imputation
    if not (in_place):
        data_old = deepcopy(data)
    _check_missing_auxvars(data, aux_vars)
    _add_impute_col(data, imp_var)
    _assign_igroups(data, aux_vars.keys())
    _calc_distances(data, aux_vars)
    _calc_donors(data, ratio=ratio)
    imputed_vals = list(
        map(
            lambda i: _impute_igroup(
                data,
                _freq_to_exp(
                    data, _get_freq_dist(data, imp_var, possible_vals, i), i),
                possible_vals,
                i,
            ),
            range(1 + data["_IGroup"].max()),
        ))
    data[imp_var + "_imputed"] = data.apply(
        lambda r: (imputed_vals[r["_IGroup"]].pop(0)
                   if imputed_vals[r["_IGroup"]] != [] else r[imp_var])
        if r["_impute"] else r[imp_var],
        axis=1,
    )
    assert all(map(lambda l: l == [], imputed_vals))
    if not (keep_intermediates):
        del data["_impute"]
        del data["_IGroup"]
        del data["_distances"]
        del data["_donor"]
    if not (in_place):
        data_new = deepcopy(data)
        data = deepcopy(data_old)
        del data_old
        return data_new

import numpy as np
import pandas as pd
from functools import reduce
from operator import add
from fractions import Fraction
from numpy.random import choice
from random import shuffle
from copy import deepcopy
from numbers import Number


class RBEISInputException(Exception):
    pass


def _add_impute_col(data, imp_var):
    """
    _add_impute_col(data, imp_var)

    data (pd.DataFrame): The dataset undergoing imputation
          imp_var (str): The name of the variable to be imputed

    Prepare the DataFrame by adding a boolean column 'impute' indicating whether
    or not a record is to be imputed.
    """
    data["_impute"] = np.isnan(data[imp_var])


def _assign_igroups(data, aux_vars):
    """
    _assign_igroups(data, aux_vars)

    data (pd.DataFrame): The dataset undergoing imputation
    aux_vars (str list): The names of the chosen auxiliary variables

    Add a column 'IGroup' containing integers representing the IGroup each
    recipient record is assigned to:
    1. Add a column '_conc_aux' containing the string representation of the list
       of values corresponding to the record's auxiliary variables
    2. Group by '_conc_aux' and extract group integers from the internal grouper
       object, creating a new column '_IGroup' containing these integers
    3. Subtract 1 from each IGroup value, giving -1 for all non-recipient
       records and zero-indexing all others
    4. Remove column '_conc_aux'
    """
    data["_conc_aux"] = data.apply(
        lambda r: str(list(map(lambda v: r[v], aux_vars)))
        if r["_impute"] else "",
        axis=1,
    )
    data["_IGroup"] = data.groupby("_conc_aux").grouper.group_info[0]
    data["_IGroup"] = data.apply(lambda r: r["_IGroup"] - 1, axis=1)
    del data["_conc_aux"]


# Distance functions
# Where x is the IGroup's value, y is the current record's value, and m is a threshold value
_df1 = lambda x, y: int(x != y)
_df2 = lambda x, y, m: int(abs(x - y) > m)
_df3 = lambda x, y, m: 1 if abs(x - y) > m else abs(x - y) / (m + 1.0)


def _get_igroup_aux_var(data, aux_var):
    """
    _get_igroup_aux_var(data, aux_var)

    data (pd.DataFrame): The dataset undergoing imputation
          aux_var (str): The desired auxiliary variable

    Return a list containing each IGroup's value of a given auxiliary variable.
    """
    out = []
    for i in range(1 + data["_IGroup"].max()):
        out += (data[[
            "_IGroup", aux_var
        ]].drop_duplicates().query("_IGroup == " +
                                   str(i))[aux_var].values.tolist())
    return out


def _build_custom_df(dist_func, mappings, threshold=None):
    """
    _build_custom_df(dist_func, mappings)

                        dist_func (int): Which of the six standard distance
                                         functions to use (in this case, only 4,
                                         5 or 6 are permissible)
    mappings (('a * 'a) * numeric dict): A dictionary with pairs of values to be
                                         overridden as its keys, and the desired
                                         distances as the corresponding values.
                                         Note that (x,y) -> z does not imply
                                         that (y,z) -> z.
                    threshold (numeric): [Optional] A threshold value for the
                                         distance function, if required

    Return a two-argument function that corresponds to distance functions 1, 2
    or 3, but with certain pairs of values overridden.  This function assumes
    that df(x,y)==df(y,x), where df is the distance function.
    """
    assert dist_func >= 4 and dist_func <= 6
    if dist_func == 4:
        df = _df1
    elif dist_func == 5:
        df = _df2
    elif dist_func == 6:
        df = _df3
    else:
        raise Exception(
            "You may only choose 4, 5 or 6 for a custom distance function.")
    if dist_func == 4:
        return lambda x, y: mappings[
            (x, y)] if (x, y) in mappings.keys() else df(x, y)
    else:
        assert threshold
        return (lambda x, y: mappings[(x, y)]
                if (x, y) in mappings.keys() else df(x, y, threshold))


def _calc_distances(data,
                    aux_vars,
                    dist_func,
                    weights,
                    threshold=None,
                    custom_df_map=None):
    """
    _calc_distances(data, aux_vars, dist_func, weights, threshold=None,
                    custom_df_map=None)

                         data (pd.DataFrame): The dataset undergoing imputation
                         aux_vars (str list): The names of the chosen auxiliary
                                              variables
                             dist_func (int): Which of the six standard distance
                                              functions to use
                weights (str * numeric dict): Dictionary with auxiliary variable
                                              names as keys and chosen weights
                                              as values
                         threshold (numeric): [Optional] A threshold value for
                                              the distance function, if required
    custom_df_map (('a * 'a) * numeric dict): [Optional] A dictionary mapping
                                              pairs (2-tuples) of values to
                                              distances, overriding the
                                              underlying standard distance
                                              function.  Note that (x,y) -> z
                                              does not imply that (y,z) -> z.

    Add a column '_distances' containing lists of calculated distances of each
    record's auxiliary variables from those of its IGroup:
    1. Create a list of dictionaries containing the values of each IGroup's
       auxiliary variables
    2. Add a column '_dists_temp' containing, for each potential donor record, a
       list of dictionaries of calculated distances for each auxiliary variable
    3. Add a column '_distances' containing, for each potential donor record, a
       list of calculated distances from its IGroup's auxiliary variables,
       taking into account the specified weights
    4. Remove column '_dists_temp'
    """
    # Calculate the distances
    dist_func = (_build_custom_df(
        dist_func, custom_df_map, threshold=threshold)
                 if dist_func >= 4 else [_df1, _df2, _df3][dist_func - 1])
    # TODO: Check if dist_func is compatible with each auxvar dtype
    #       (e.g. DF3 doesn't like strings)
    igroup_aux_vars = []
    vars_vals = list(
        zip(aux_vars, map(lambda v: _get_igroup_aux_var(data, v), aux_vars)))
    for i in range(1 + data["_IGroup"].max()):
        igroup_aux_vars.append({k: v[i] for k, v in vars_vals})
    data["_dists_temp"] = data.apply(
        lambda r: list(
            map(
                lambda x:
                {k: weights[k] * dist_func(x[k], r[k])
                 for k in aux_vars},
                igroup_aux_vars,
            )) if not (r["_impute"]) else [],
        axis=1,
    )
    data["_distances"] = data.apply(
        lambda r: list(
            map(lambda d: reduce(add, list(d.values()), 0), r["_dists_temp"])),
        axis=1,
    )
    del data["_dists_temp"]


def _calc_donors(data, min_quantile=None):
    """
    _calc_donors(data, min_quantile=None)

    data (pd.DataFrame): The dataset undergoing imputation
     min_quantile (int): [Optional] Instead of choosing the minimum distance,
                         choose records in the lowest n-quantile, where
                         min_quantile specifies n (e.g. if min_quantile is 10,
                         then accept records in the bottom decile).

    Add a column 'donor' containing a list of IGroup numbers to which each
    record is a donors.
    1. Calculate the distances less than or equal to which a record may be
       accepted as a donor to each respective IGroup.
    2. Zip each record's distances to the list of distances calculated in step
       1, and identify where the record distances are less than or equal to the
       maximum required IGroup distances, giving a list of indices where this is
       the case.
    """
    igroups_dists = np.array(
        data.query("not(_impute)")["_distances"].values.tolist()).T.tolist()
    for i in igroups_dists:
        i.sort()
    # This would be a lot nicer if we could have NumPy >= 1.15.0, which has
    # np.quantile, but we're on 1.13.3
    max_donor_dists = list(
        map(
            (lambda l: l[int(np.ceil(len(l) / min_quantile)) - 1])
            if min_quantile else min,
            igroups_dists,
        ))
    data["_donor"] = data.apply(
        lambda r: np.where(
            list(
                map(lambda p: p[0] <= p[1],
                    zip(r["_distances"], max_donor_dists))))[0].tolist()
        if not (r["_impute"]) else [],
        axis=1,
    )


def _get_donors(data, igroup):
    """
    _get_donors(data, igroup)

    data (pd.DataFrame): The dataset undergoing imputation
           igroup (int): The IGroup whose donors are required

    Return a list of indices corresponding to records in data that are donors to
    the specified IGroup.
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
    proportions of a given igroup taken up by each possible value.
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
    weights,
    dist_func,
    threshold=None,
    custom_df_map=None,
    min_quantile=None,
    overwrite=False,
    col_name=None,
    in_place=True,
    keep_intermediates=False,
):
    """
    impute(data, imp_var, possible_vals, aux_vars, weights, dist_func,
           threshold=None, custom_df_map=None, min_quantile=None,
           overwrite=False, col_name=None, in_place=True,
           keep_intermediates=False)

                         data (pd.DataFrame): The dataset undergoing imputation
                               imp_var (str): The name of the variable to be
                                              imputed
                     possible_vals ('a list): A list of all possible values that
                                              imp_var can take
                         aux_vars (str list): The names of the chosen auxiliary
                                              variables
                weights (str * numeric dict): Dictionary with auxiliary variable
                                              names as keys and chosen weights
                                              as values
                             dist_func (int): Which of the six standard distance
                                              functions to use
                         threshold (numeric): [Optional] A threshold value for
                                              the distance function, if required
    custom_df_map (('a * 'a) * numeric dict): [Optional] A dictionary mapping
                                              pairs (2-tuples) of values to
                                              distances, overriding the
                                              underlying standard distance
                                              function.  Note that (x,y) -> z
                                              does not imply that (y,z) -> z.
                          min_quantile (int): [Optional] Instead of choosing the
                                              minimum distance, choose records
                                              in the lowest n-quantile, where
                                              min_quantile specifies n (e.g. if
                                              min_quantile is 10, then accept
                                              records in the bottom decile).
                            overwrite (bool): [Optional, default False] True if
                                              the column to undergo imputation
                                              should be overwritten with the
                                              results, False if imputed values
                                              are to go into a new column.
                              col_name (str): [Optional] If overwrite is False,
                                              the name of the new column in
                                              which to write the imputed values
                             in_place (bool): [Optional, default True] If True,
                                              modify the original DataFrame in
                                              place.  If False, return a new
                                              (deep) copy of the DataFrame
                                              having undergone imputation.
                   keep_intermediates (bool): [Optional, default False] If True,
                                              retain the intermediate columns
                                              created by this implementation of
                                              RBEIS in the process of
                                              imputation.  If False,
                                              remove them from the output.

    Impute missing values for a given variable using the Rogers & Berriman
    Editing and Imputation System (RBEIS).  A high-level overview of the
    approach is given here (for more detail, see the documentation for each of
    the intermediate functions in this library):
    1. Identify values to be imputed
    2. Assign imputation groups ("IGroups") based on a given set of auxiliary
       variables
    3. Calculate how similar the auxiliary variables of each IGroup are to those
       of the potential donor records
    4. Assign the most similar records to the donor pools of the corresponding
       IGroups
    5. Impute values for each IGroup
    6. Insert imputed values into the original DataFrame
    """

    # Input checks
    if not (isinstance(data, pd.DataFrame)):
        raise TypeError("Dataset must be a Pandas DataFrame")
    if not (isinstance(imp_var, str)):
        raise TypeError("Imputation variable name must be a string")
    if not (isinstance(possible_vals, list)):
        raise TypeError("Possible values must be contained in a list")
    if not (isinstance(aux_vars, list)):
        raise TypeError("Auxiliary variable names must be a list of strings")
    if not (all(map(lambda x: isinstance(x, str), aux_vars))):
        raise TypeError("Auxiliary variable names must be a list of strings")
    if not (isinstance(weights, dict)):
        raise TypeError(
            "Weights must be a dictionary whose keys are strings containing variable names and whose values are numeric"
        )
    if not (all(map(lambda x: isinstance(x, str), weights.keys()))):
        raise TypeError(
            "Weights must be a dictionary whose keys are strings containing variable names"
        )
    if not (all(map(lambda x: isinstance(x, Number), weights.values()))):
        raise TypeError(
            "Weights must be a dictionary whose values are numeric")
    try:
        assert all(map(lambda v: v in weights.keys(), aux_vars))
    except AssertionError:
        raise RBEISInputException(
            "You have not specified a weight for every auxiliary variable")
    if not (isinstance(dist_func, int)):
        raise TypeError(
            "Distance functions are indicated by an integer from 1 to 6 inclusive"
        )
    if dist_func < 1 or dist_func > 6:
        raise RBEISInputException(
            "The distance function must be an integer from 1 to 6 inclusive")
    if dist_func in [2, 3, 5, 6]:
        try:
            assert threshold
            if not (isinstance(threshold, Number)):
                raise TypeError(
                    "Distance function thresholds must be a numeric type")
        except AssertionError:
            raise RBEISInputException(
                "The chosen distance function requires a threshold value")
    if dist_func >= 4:
        try:
            assert custom_df_map
            if not (all(
                    map(lambda x: isinstance(x, tuple),
                        custom_df_map.keys()))):
                raise TypeError(
                    "Distance function overrides must be expressed in a dictionary whose keys are tuples representing pairs of values"
                )
            if not (all(
                    map(lambda x: isinstance(x, Number),
                        custom_df_map.values()))):
                raise TypeError(
                    "Distance function overrides must be expressed in a dictionary whose values are numeric"
                )
        except AssertionError:
            raise RBEISInputException(
                "You have chosen a distance funtion with overrides, but have not provided them"
            )
    try:
        assert min_quantile
        if not (isinstance(min_quantile, int)):
            raise TypeError(
                "The required quantile partition must be specified with an integer"
            )
    except AssertionError:
        pass
    try:
        assert overwrite
        if not (isinstance(overwrite, bool)):
            raise TypeError("overwrite must be either True or False")
    except AssertionError:
        pass
    try:
        assert col_name
        if not (isinstance(col_name, str)):
            raise TypeError("col_name must be a string")
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
    _add_impute_col(data, imp_var)
    _assign_igroups(data, aux_vars)
    _calc_distances(
        data,
        aux_vars,
        dist_func,
        weights,
        threshold=threshold,
        custom_df_map=custom_df_map,
    )
    _calc_donors(data, min_quantile=min_quantile)
    # TODO: tidy this up and return the dataframe properly (or modify in place)
    # TODO: include optional keyword arguments
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

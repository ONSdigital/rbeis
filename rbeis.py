import numpy as np
import pandas as pd
from functools import reduce
from operator import add
from fractions import Fraction
from numpy.random import choice
from random import shuffle
from copy import deepcopy


def _add_impute_col(data, imp_var):
    """
    data (pd.DataFrame): The dataset undergoing imputation
          imp_var (str): The name of the variable to be imputed

    Prepare the DataFrame by adding a boolean column 'impute' indicating whether
    or not a record is to be imputed.
    """
    data["impute"] = np.isnan(data[imp_var])


def _assign_igroups(data, aux_vars):
    """
    data (pd.DataFrame): The dataset undergoing imputation
    aux_vars (str list): The names of the chosen auxiliary variables

    Add a column 'IGroup' containing integers representing the IGroup each
    recipient record is assigned to:
    1. Add a column 'conc_aux' containing the string representation of the list
       of values corresponding to the record's auxiliary variables
    2. Group by 'conc_aux' and extract group integers from the internal grouper
       object, creating a new column 'IGroup' containing these integers
    3. Subtract 1 from each IGroup value, giving -1 for all non-recipient
       records and zero-indexing all others
    4. Remove column 'conc_aux'
    """
    data["conc_aux"] = data.apply(
        lambda r: str(list(map(lambda v: r[v], aux_vars))) if r["impute"] else "",
        axis=1,
    )
    data["IGroup"] = data.groupby("conc_aux").grouper.group_info[0]
    data["IGroup"] = data.apply(lambda r: r["IGroup"] - 1, axis=1)
    del data["conc_aux"]


# Distance functions
# Where x is the IGroup's value, y is the current record's value, and m is a threshold value
_df1 = lambda x, y: int(x != y)
_df2 = lambda x, y, m: int(abs(x - y) > m)
_df3 = lambda x, y, m: 1 if abs(x - y) > m else abs(x - y) / (m + 1)


def _get_igroup_aux_var(data, aux_var):
    """
    data (pd.DataFrame): The dataset undergoing imputation
          aux_var (str): The desired auxiliary variable

    Return a list containing each IGroup's value of a given auxiliary variable.
    """
    out = []
    for i in range(1 + data["IGroup"].max()):
        out += (
            data[["IGroup", aux_var]]
            .drop_duplicates()
            .query("IGroup == " + str(i))[aux_var]
            .values.tolist()
        )
    return out


def _build_custom_df(dist_func, mappings):
    """
                        dist_func (int): Which of the six standard distance
                                         functions to use (in this case, only 4,
                                         5 or 6 are permissible)
    mappings (('a * 'a) * numeric dict): A dictionary with pairs of values to be
                                         overridden as its keys, and the desired
                                         distances as the corresponding values

    Return a two-argument function that corresponds to distance functions 1, 2
    or 3, but with certain pairs of values overridden.
    """
    assert dist_func >= 4 and dist_func <= 6
    if dist_func == 4:
        df = _df1
    elif dist_func == 5:
        df = _df2
    elif dist_func == 6:
        df = _df3
    else:
        raise Exception("You may only choose 4, 5 or 6 for a custom distance function.")
    return lambda x, y: mappings[(x, y)] if (x, y) in mappings.keys() else df(x, y)


def _calc_distances(
    data, aux_vars, dist_func, weights, threshold=None, custom_df_map=None
):
    """
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
                                              pairs of values to distances,
                                              overriding the underlying standard
                                              distance function

    Add a column 'distances' containing lists of calculated distances of each
    record's auxiliary variables from those of its IGroup:
    1. Create a list of dictionaries containing the values of each IGroup's
       auxiliary variables
    2. Add a column 'dists_temp' containing, for each potential donor record, a
       list of dictionaries of calculated distances for each auxiliary variable
    3. Add a column 'distances' containing, for each potential donor record, a
       list of calculated distances from its IGroup's auxiliary variables,
       taking into account the specified weights
    4. Remove column 'dists_temp'
    """
    # Check that the distance function selected is one of the standard six
    assert dist_func >= 1 and dist_func <= 6
    # If required, check that there is a threshold
    if dist_func == 3:
        assert threshold
    # If DF4, DF5 or DF6 required, check that map is provided
    if dist_func >= 4:
        assert custom_df_map
    # Check that all of the auxiliary variables have weights specified
    assert all(map(lambda v: v in aux_vars, weights.keys()))
    # Calculate the distances
    dist_func = (
        _build_custom_df(dist_func, custom_df_map)
        if dist_func >= 4
        else [_df1, _df2, _df3][dist_func - 1]
    )
    igroup_aux_vars = []
    vars_vals = list(
        zip(aux_vars, map(lambda v: _get_igroup_aux_var(data, v), aux_vars))
    )
    for i in range(1 + data["IGroup"].max()):
        igroup_aux_vars.append({k: v[i] for k, v in vars_vals})
    data["dists_temp"] = data.apply(
        lambda r: list(
            map(
                lambda x: {k: weights[k] * dist_func(x[k], r[k]) for k in aux_vars},
                igroup_aux_vars,
            )
        )
        if not (r["impute"])
        else [],
        axis=1,
    )
    data["distances"] = data.apply(
        lambda r: list(
            map(lambda d: reduce(add, list(d.values()), 0), r["dists_temp"])
        ),
        axis=1,
    )
    del data["dists_temp"]


def _calc_donors(data, min_quantile=None):
    """
    data (pd.DataFrame): The dataset undergoing imputation
     min_quantile (int): Instead of choosing the minimum distance, choose
                         records in the lowest n-quantile, where min_quantile
                         specifies n (e.g. if min_quantile is 10, then accept
                         records in the bottom decile).

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
        data.query("not(impute)")["distances"].values.tolist()
    ).T.tolist()
    igroups_dists.sort()
    # This would be a lot nicer if we could have NumPy >= 1.15.0, which has
    # np.quantile, but we're on 1.13.3
    max_donor_dists = list(
        map(
            (lambda l: l[int(np.ceil(len(l) / n)) - 1]) if min_quantile else min,
            igroups_dists,
        )
    )
    data["donor"] = data.apply(
        lambda r: np.where(
            list(map(lambda p: p[0] <= p[1], zip(r["distances"], max_donor_dists)))
        )[0].tolist()
        if not (r["impute"])
        else [],
        axis=1,
    )


def _get_donors(data, igroup):
    """
    data (pd.DataFrame): The dataset undergoing imputation
           igroup (int): The IGroup whose donors are required

    Return a list of indices corresponding to records in data that are donors to
    the specified IGroup.
    """
    return list(map(lambda x: igroup in x, data["donor"].values.tolist()))


def _get_freq_dist(data, imp_var, possible_vals, igroup):
    """
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
        map(lambda n: 0 if n==0 else Fraction(n,len(pool)),
        map(
            lambda v: len([i for i in pool if i == v]),
            possible_vals,
        ))
    )

def _freq_to_exp(freq_dist,igroup):
    """
    freq_dist (Fraction list): The frequency distribution to convert
                 igroup (int): The IGroup corresponding to freq_dist

    Convert a frequency distribution to the expected numbers of occurrences for
    a given IGroup.
    """
    igroup_size = len(data.query("IGroup=="+str(igroup)).values.tolist())
    return list(map(lambda f: f*igroup_size,freq_dist))


def impute(
    data,
    imp_var,
    possible_vals,
    aux_vars,
    weights,
    dist_func,
    min_quantile=None,
    overwrite=False,
    col_name=None,
):
    # data: dataframe
    # imp_var: string
    # possible_vals: 'a list
    # aux_vars: string list
    # weights: (string * float) dict
    # dist_func: numeric * numeric -> numeric function
    # min_quantile: int
    # overwrite: bool
    # col_name: string
    # - (check that weights keys correspond to aux_vars)
    # - add 'impute' column
    # - assign igroups
    # - calculate distances
    # - get donors
    # - get igroup freq dists
    # - convert to rbeis donor pools
    # - assign imputed variables
    # - return dataset
    pass


# Test setup: same dataset as in the notebook example
test_data = pd.read_csv("../understanding/data.csv")
test_imp_var = "white"
test_aux_vars = ["interim_id", "gor9d", "work_status_group", "dvhsize"]
test_df = 1
test_weights = {k: 1.0 for k in test_aux_vars}


def t1():
    _add_impute_col(test_data, test_imp_var)


def t2():
    _assign_igroups(test_data, test_aux_vars)


def t3():
    _calc_distances(test_data, test_aux_vars, test_df, test_weights)


def t4():
    _calc_donors(test_data)

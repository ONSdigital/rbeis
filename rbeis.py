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
    3. Assign non-recipient records to IGroup -1
    4. Remove column 'conc_aux'
    """
    data["conc_aux"] = data.apply(
        lambda r: str(list(map(lambda v: r[v], aux_vars))) if r["impute"] else "",
        axis=1,
    )
    data["IGroup"] = data.groupby("conc_aux").grouper.group_info[0]
    data["IGroup"] = np.where(data["impute"] == False, -1, data["IGroup"])
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
    for i in range(1, 1 + data["IGroup"].max()):
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
    assert dist_func >=4 and dist_func <= 6
    if dist_func==4: df=_df1
    elif dist_func==5: df=_df2
    elif dist_func==6: df=_df3
    else: raise Exception("You may only choose 4, 5 or 6 for a custom distance function.")
    return lambda x,y: mappings[(x,y)] if (x,y) in mappings.keys() else df(x,y)

def _calc_distances(data, aux_vars, dist_func, weights, threshold=None, custom_df_map=None):
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
    dist_func = _build_custom_df(dist_func,custom_df_map) if dist_func>=4 else [_df1, _df2, _df3][dist_func - 1]
    igroup_aux_vars = []
    vars_vals = zip(aux_vars, map(lambda v: _get_igroup_aux_var(data, v), aux_vars))
    for i in range(0, data["IGroup"].max()):
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


def _get_donors(igroup):
    pass


def _get_freq_dist(igroup):
    pass


def _freq_to_exp(freq_dist):
    pass


def impute(
    data, impute_var, aux_vars, weights, dist_func, min_margin, overwrite, col_name=None
):
    # data: dataframe
    # impute_var: string
    # aux_vars: string list
    # weights: (string * float) dict
    # dist_func: numeric * numeric -> numeric function
    # min_margin: float
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
test_weights = {k: 1.0 for k in test_aux_vars}


def t1():
    _add_impute_col(test_data, test_imp_var)


def t2():
    _assign_igroups(test_data, test_aux_vars)


def t3():
    _calc_distances(test_data, test_aux_vars, 1, test_weights)

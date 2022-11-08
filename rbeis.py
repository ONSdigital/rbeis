import numpy as np
import pandas as pd
from functools import reduce
from operator import add
from fractions import Fraction
from numpy.random import choice
from random import shuffle
from copy import deepcopy

# Test setup: same dataset as in the notebook example
test_data = pd.read_csv("../understanding/data.csv")
impvar = "white"
auxvars = ["interim_id", "gor9d", "work_status_group", "dvhsize"]

def _prep(data):
    pass

def _assign_igroups(data, aux_vars):
    pass

def _calc_distances(data, dist_func):
    pass

def _get_donors(igroup):
    pass

def _get_freq_dist(igroup):
    pass

def _freq_to_exp(freq_dist):
    pass

def impute(*, data, impute_var, aux_vars, weights, dist_func, min_margin, overwrite):
    # data: dataframe
    # impute_var: string
    # aux_vars: string list
    # weights: (string * float) dict
    # dist_func: numeric * numeric -> numeric function
    # min_margin: float
    # overwrite: bool
    # - (check that weights keys correspond to aux_vars)
    # - prep data
    # - assign igroups
    # - calculate distances
    # - get donors
    # - get igroup freq dists
    # - convert to rbeis donor pools
    # - assign imputed variables
    # - return dataset
    pass

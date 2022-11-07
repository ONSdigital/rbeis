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


def impute(*, data: int, impute_var: int, aux_vars: int) -> int:
    return data + impute_var + aux_vars

t = [1,2,3]

from unittest import TestCase

import numpy as np
import pandas as pd
from fractions import Fraction

from rbeis.pandas import (
    RBEISDistanceFunction,
    impute,
    _add_impute_col,
    _assign_igroups,
    _get_igroup_aux_var,
    _calc_distances,
    _calc_donors,
    _get_donors,
    _get_freq_dist,
    _freq_to_exp,
    _impute_igroup,
)


class RBEISTestCase(TestCase):

    # --- Set up simple dummy dataframe for testing input parameters for impute function  ---
    dummy_data = {
        "dummy_impute_var": [30, 20, 30, 40, None],
        "dummy_aux_var1": [21, 19, 18, 67, 20],
        "dummy_aux_var2": [2, 3, 1, 1, 3],
        "dummy_aux_var_missing": [4, 3, 6, None, 6],
        "dummy_aux_var_categorical": ['dog', 'cat', 'parrot', 'giraffe', 'hedgehog'] 
    }
    dummy_dataframe = pd.DataFrame(dummy_data)

    # --- Set up variables for importing test data ---
    test_data_filepath = "tests/artists_unique_galcount_spaceavg_missing.csv"
    test_impute_var = "whitney_count"
    test_pos_vals = list(range(0, 41))
    test_aux_var1 = "moma_count"
    test_threshold1 = 2
    test_custom_map1 = {(1, 0): 0.5}
    test_aux_var2 = "space_ratio_per_page_avg"
    test_threshold2 = 0.2
    test_aux_var_categorial = "artist_nationality_other"
    test_custom_map_categorial = {("British", "American"): 2}

    # Set up data for testing distances calulations
    # For IGroup 0, moma_count = 0,
    # space_ratio_per_page_avg = 0.197291
    # & artist_nationality_other = "American"
    igroup0_aux_var1 = 0
    igroup0_aux_var2 = 0.197291
    igroup0_aux_categorical = "American"


# =============================================================================
# ----------------------      TESTING TEMPLATE      ---------------------------
# =============================================================================
# - Test type validation on all input parameters
# - Test constraints on input parameters:
# --     impute variable and auxiliary variables are in the dataframe
# --     auxiliry variables have no missing values
# --     weights and distance functions are within range
# --     threshold & custom_map are specified for, and only for, appropriate
#                                                          distance function
# - Test each function does what it is expected to do
# - Test if output dataframe is as expected, both new columns and data content

# TEST DATA:
# Data is loaded for each function test as the order tests are carried out in
# cannot be controlled
# =============================================================================


# -----------------------------------------------------------------------------
# TESTS: TYPE VALIDATION ON INPUT PARAMETERS
# -----------------------------------------------------------------------------
class TestInputTypeValidation(RBEISTestCase):

   # --- Test type validation on the input data (dataframe)  ---
    def test_type_validation_data(self):
        with self.assertRaises(TypeError):
            impute(
                data=["Not_a_Dataframe"],
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test type validation on the impute variable (string) ---
    def test_type_validation_imp_var(self):
        with self.assertRaises(TypeError):
            impute(
                data=self.dummy_dataframe,
                imp_var=["Not", "A", "String"],
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test type validation on possible values for impute variable (list) ---
    def test_type_validation_possible_vals(self):
        with self.assertRaises(TypeError):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals="Not_a_List",
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test type validation on auxiliary variables (dictionary) ---
    def test_type_validation_aux_vars(self):
        with self.assertRaises(TypeError):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars="Not_a_Dictionary",
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test type validation on auxiliary variables dictionary keys (string) ---
    def test_type_validation_aux_var_keys(self):
        with self.assertRaises(TypeError):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    ["Not", "A", "String"]:
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test type validation on RBEISDistanceFunction (RBEISDistanceFunction) ---
    def test_type_validation_dist_func(self):
        with self.assertRaises(TypeError):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    "Not_an_RBEISDistanceFunction",
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test type validation on distance function number ---
    def test_type_validation_df(self):
        with self.assertRaises(TypeError):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction("Not_a_Number",
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(4,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test type validation on custom_map (dictionary) ---
    def test_type_validation_custom_map(self):
        with self.assertRaises(TypeError):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(4,
                                          custom_map="Not_a_Dictionary",
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test type validation on threshhold (numeric) ---
    def test_type_validation_threshold(self):
        with self.assertRaises(TypeError):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(2,
                                          custom_map=None,
                                          threshold="Not_a_Number",
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test type validation on weights (numeric) ---
    def test_type_validation_weights(self):
        with self.assertRaises(TypeError):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight="Not_a_Number"),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test type validation on ratio  ---
    def test_type_validation_ratio(self):
        with self.assertRaises(TypeError):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio="Not_a_number",
                keep_intermediates=True,
            )

    # --- Test type validation on keep_intermediates  ---
    def test_type_validation_keep_intermediates(self):
        with self.assertRaises(TypeError):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates="Not_a_Boolean",
            )

# -----------------------------------------------------------------------------
# TESTS: CHECK VALUES OF INPUT VARIABLES 
# -----------------------------------------------------------------------------
class TestInputValues(RBEISTestCase):

    # --- Test exception when impute variable is not in the dataframe  ---
    def test_imp_var_in_df(self):
        with self.assertRaises(Exception):
            impute(
                data=self.dummy_dataframe,
                imp_var="not_in_dataframe",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test exception when possible_vals do not match range of impute variable  ---
    # def test_pos_vals_match_range(self):
    #    with self.assertRaises(Exception):
    #        impute(
    #          data=self.dummy_dataframe,
    #          imp_var="dummy_impute_var",
    #          possible_vals=list(range(100,120)),
    #            aux_vars={
    #                    "dummy_aux_var1":
    #                        RBEISDistanceFunction(
    #                        1,
    #                        custom_map=None,
    #                        threshold=None,
    #                        weight=2),
    #                    "dummy_aux_var2":
    #                        RBEISDistanceFunction(
    #                        1,
    #                        custom_map=None,
    #                        threshold=None,
    #                        weight=3)
    #                    },
    #          ratio=None,
    #          keep_intermediates=True)

    # --- Test exception when auxiliary variable is not a variable in the dataframe
    def test_aux_vars_in_df(self):
        with self.assertRaises(Exception):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "not_in_dataframe":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test exception when auxiliary variable has missing values
    def test_aux_vars_no_missing_vals(self):
        with self.assertRaises(Exception):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var_missing":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test exception when weights are not positive values
    def test_weights_are_positive(self):
        with self.assertRaises(Exception):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=-1),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test exception when distance function specified is not int between 1 & 6
    def test_dist_func_1_to_6(self):
        with self.assertRaises(Exception):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(7,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

     # --- Test exception when distance function 2,3,5 or 6 is specified when
     #      auxiliary variable is categorical (only df 1 or 4 allowed)
    def test_df_for_categorical_data(self):
        with self.assertRaises(Exception):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                        "dummy_aux_var1": 
                        RBEISDistanceFunction(1, 
                                              custom_map=None, 
                                              threshold=None, 
                                              weight=2), 
                        "dummy_aux_var_categorical": 
                        RBEISDistanceFunction(3, 
                                              custom_map=None, 
                                              threshold=2, 
                                              weight=3)
                },
                ratio=None,
                keep_intermediates=True,
            )     

    # --- Test warning is given when threshold IS specified for distance function 1
    def test_threshold_for_df1(self):
        with self.assertWarns(UserWarning):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          threshold=3,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test exception when threshold NOT specified for distance function 2
    def test_threshold_for_df2(self):
        with self.assertRaises(Exception):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(2,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test exception when threshold NOT specified for distance function 3
    def test_threshold_for_df3(self):
        with self.assertRaises(Exception):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(3,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test warning is given when threshold IS specified for distance function 4
    def test_threshold_for_df4(self):
        with self.assertWarns(UserWarning):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(4,
                                          custom_map={(1, 1): 2},
                                          threshold=3,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test exception when threshold NOT specified for distance function 5
    def test_threshold_for_df5(self):
        with self.assertRaises(Exception):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(5,
                                          custom_map={(1, 1): 2},
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test exception when threshold NOT specified for distance function 6
    def test_threshold_for_df6(self):
        with self.assertRaises(Exception):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(6,
                                          custom_map={(1, 1): 2},
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test warning is given when custom_map IS specified for distance function 1
    def test_custom_map_for_df1(self):
        with self.assertWarns(UserWarning):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          custom_map={(1, 1): 2},
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test warning is given when custom_map IS specified for distance function 2
    def test_custom_map_for_df2(self):
        with self.assertWarns(UserWarning):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(2,
                                          custom_map={(1, 1): 2},
                                          threshold=1,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test warning is given when custom_map IS specified for distance function 3
    def test_custom_map_for_df3(self):
        with self.assertWarns(UserWarning):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(3,
                                          custom_map={(1, 1): 2},
                                          threshold=1,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test exception when custom_map NOT specified for distance function 4
    def test_custom_map_for_df4(self):
        with self.assertRaises(Exception):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(4,
                                          custom_map=None,
                                          threshold=None,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test exception when custom_map NOT specified for distance function 5
    def test_custom_map_for_df5(self):
        with self.assertRaises(Exception):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(5,
                                          custom_map=None,
                                          threshold=1,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test exception when custom_map NOT specified for distance function 6
    def test_custom_map_for_df6(self):
        with self.assertRaises(Exception):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1":
                    RBEISDistanceFunction(1,
                                          custom_map=None,
                                          threshold=None,
                                          weight=2),
                    "dummy_aux_var2":
                    RBEISDistanceFunction(6,
                                          custom_map=None,
                                          threshold=1,
                                          weight=3),
                },
                ratio=None,
                keep_intermediates=True,
            )

    # --- Test exception when ratio is NOT greater than 1
    def test_ratio_greater1(self):
        with self.assertRaises(Exception):
            impute(
                data=self.dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars={
                    "dummy_aux_var1": 
                    RBEISDistanceFunction(1, 
                                          custom_map=None, 
                                          threshold=None, 
                                          weight=2), 
                    "dummy_aux_var2": 
                    RBEISDistanceFunction(1, 
                                          custom_map=None, 
                                          threshold=None, 
                                          weight=3)
                },              
                ratio=0.5,
                keep_intermediates=True,
            )


# --------------------------------------------------------------------------------------
# TESTS: DISTANCE FUNCTIONS: RBEISDistanceFunction Class
#
# It is assumed that :
# 1. missing values cannot be passed to the distance functions
#    as imputation will be prevented when the impute function is called
#    if any of the auxiliary values contain missing values.
# 2. the df number will be between 1 & 6 as an exeption will be
#    raised when the impute function is called if not
# --------------------------------------------------------------------------------------
class TestDistanceFunctions(RBEISTestCase):

    # --- Test distance function 1 returns correct values ---
    def test_df1(self):
        df1 = RBEISDistanceFunction(1)
        self.assertEqual(df1(1, 1), 0)
        self.assertEqual(df1("abcd", "abcd"), 0)
        self.assertEqual(df1(0, 1), 1)
        self.assertEqual(df1(2.3, 1.2), 1)
        self.assertEqual(df1("abcd", "bcde"), 1)

    # --- Test distance function 2 returns correct values ---
    def test_df2(self):
        df2 = RBEISDistanceFunction(2, threshold=3)
        self.assertEqual(df2(1, 1), 0)
        self.assertEqual(df2(1, 5), 1)
        df2 = RBEISDistanceFunction(2, threshold=3.0)
        self.assertEqual(df2(1.1, 3.5), 0)
        df2 = RBEISDistanceFunction(2, threshold=3.5)
        self.assertEqual(df2(7.8, 2.1), 1)

    # --- Test distance function 3 returns correct values ---
    def test_df3(self):
        df3 = RBEISDistanceFunction(3, threshold=3)
        self.assertEqual(df3(1, 1), 0)
        self.assertEqual(df3(1, 3), 0.5)
        self.assertEqual(df3(7, 6), 0.25)
        self.assertEqual(df3(7, 2), 1)

    # --- Test distance function 4 returns correct values ---
    def test_df4(self):
        df4 = RBEISDistanceFunction(4, custom_map={(1, 1): 2, (2.2, 3.3): 8})
        self.assertEqual(df4(2.2, 3.3), 8.0)
        self.assertEqual(df4(2, 3), 1)
        self.assertEqual(df4(2, 2), 0)
        self.assertEqual(df4("abcd", "abcd"), 0)
        self.assertEqual(df4(1.1, 2.2), 1)
        self.assertEqual(df4("abcd", "bcde"), 1)

    # --- Test distance function 5 returns correct values ---
    def test_df5(self):
        df5 = RBEISDistanceFunction(5,
                                    custom_map={
                                        (1, 1): 2,
                                        (2, 3): 8
                                    },
                                    threshold=3)
        self.assertEqual(df5(1, 1), 2)
        self.assertEqual(df5(1, 4), 0)
        self.assertEqual(df5(1.1, 5.5), 1)
        self.assertEqual(df5(7, 2), 1)

    # --- Test distance function 6 returns correct values ---
    def test_df6(self):
        df6 = RBEISDistanceFunction(6, custom_map={(1, 1): 2}, threshold=3.0)
        self.assertEqual(df6(1, 1), 2.0)
        self.assertEqual(df6(5, 5), 0)
        self.assertEqual(df6(1, 3), 0.5)
        self.assertEqual(df6(7, 6), 0.25)
        self.assertEqual(df6(7, 2), 1)


# -----------------------------------------------------------------------------
# TESTS: Function: _add_impute_col
# This function adds a boolean column 'impute' indicating whether record
# will be imputed.
# -----------------------------------------------------------------------------
class TestAddImputeCol(RBEISTestCase):

    # --- Test impute column values are assigned correctly ---
    def test_assign_impute(self):
        test_data = pd.read_csv(self.test_data_filepath)
        _add_impute_col(test_data, self.test_impute_var)

        # Check that correct Boolean values are assigned for _impute column
        self.assertTrue(test_data["_impute"].equals(
            np.isnan(test_data[self.test_impute_var])))


# -----------------------------------------------------------------------------
# TESTS: Function: _assign_igroups
# This function adds a column 'IGroup' containing integers representing
# the IGroup each recipient record is assigned to.
# -----------------------------------------------------------------------------
class TestAssignIgroups(RBEISTestCase):

    # --- Test Igroup column values are assigned correctly ---
    # Categorical auxiliary variable used to ensure there are some iGroups
    # with a single recipient and some with multiple recipients
    def test_assign_igroups(self):
        test_data = pd.read_csv(self.test_data_filepath)
        _add_impute_col(test_data, self.test_impute_var)
        _assign_igroups(test_data, [self.test_aux_var1, self.test_aux_var_categorial])

        # Check Igroup is set to -1 for non-recipient records
        donors = test_data[test_data["_impute"] == False]
        for row_index in range(donors.shape[0]):
            self.assertTrue(donors["_IGroup"].values[row_index] == -1)

        # Check Igroup is assigned for recipients
        recipients = test_data[test_data["_impute"] == True]
        for row_index in range(recipients.shape[0]):
            self.assertTrue(recipients["_IGroup"].values[row_index] > -1)

        # Calculate how many recipients in each IGroup
        recipient_freq = recipients.groupby(by="_IGroup")["_IGroup"].count()

        # Check records assigned to same IGroup have same valueas for
        # auxiliary variables
        multiple_recipient_igroups = recipient_freq[recipient_freq > 1].index
        
        for igroup_number in multiple_recipient_igroups:
            igroup_data = test_data[test_data["_IGroup"] == igroup_number]
            self.assertTrue(len(igroup_data[self.test_aux_var1].unique()) == 1)
            self.assertTrue(len(igroup_data[self.test_aux_var_categorial].unique()) == 1)         

        # Check single recipient IGroups all have different combination of
        # auxiliary variables
        single_recipient_list = recipient_freq[recipient_freq == 1].index
        single_recipient_data = test_data.copy()
        single_recipient_data = single_recipient_data[
            single_recipient_data["_IGroup"].isin(single_recipient_list)]

        single_recipient_data["combine_aux_vars"] = (
            single_recipient_data[self.test_aux_var1].astype(str) + " " +
            single_recipient_data[self.test_aux_var_categorial].astype(str))

        single_recipient_freq = single_recipient_data.groupby(
            by="combine_aux_vars")["combine_aux_vars"].count()

        self.assertTrue(
            len(single_recipient_freq[single_recipient_freq > 1].index) == 0)


# -----------------------------------------------------------------------------
# TESTS: Function: _get_igroup_aux_var
# This function returns a list containing each IGroup's value of a given
# auxiliary variable.
# -----------------------------------------------------------------------------
class TestGetIGroupAuxVar(RBEISTestCase):

    # --- Test list of auxiliary variables values for each iGroup is correct ---
    def test_get_igroup_aux_var(self):
        test_data = pd.read_csv(self.test_data_filepath)
        _add_impute_col(test_data, self.test_impute_var)
        _assign_igroups(test_data, [self.test_aux_var1, self.test_aux_var2])

        # Get list of auxiliary variable values for each iGroup
        aux_var1_list = _get_igroup_aux_var(test_data, self.test_aux_var1)
        aux_var2_list = _get_igroup_aux_var(test_data, self.test_aux_var2)

        # Check records assigned to each iGroup match auxiliary values in list
        for igroup_number in range(1 + test_data["_IGroup"].max()):
            for row_index in range(test_data.shape[0]):
                if test_data["_IGroup"].values[row_index] == igroup_number:
                    self.assertTrue(
                        test_data[self.test_aux_var1].values[row_index] ==
                        aux_var1_list[igroup_number])
                    self.assertTrue(
                        test_data[self.test_aux_var2].values[row_index] ==
                        aux_var2_list[igroup_number])


# -----------------------------------------------------------------------------
# TESTS: Function: _calc_distances
# This function adds a column '_distances' containing lists of calculated
# distances of each record's auxiliary variables from those of its IGroup.
# -----------------------------------------------------------------------------
class TestCalcDistances(RBEISTestCase):

    # Test distances are calulated correctly for each distance function
    # For iGroup 0 only
    # Distance function object takes parameters (donor record value, iGroup value)
    # Format for custom_map is (donor record value, iGroup value) : new distance

    # --- Test distance calculations using distance function 1  ---
    def test_calc_distances_df1(self):
        test_data = pd.read_csv(self.test_data_filepath)
        _add_impute_col(test_data, self.test_impute_var)
        _assign_igroups(test_data,
                        [self.test_aux_var1, self.test_aux_var_categorial])
        _calc_distances(
            data=test_data,
            aux_vars={
                self.test_aux_var1: RBEISDistanceFunction(1, weight=2),
                self.test_aux_var_categorial: RBEISDistanceFunction(1,
                                                                    weight=3),
            },
        )

        # Recalculate distances for iGroup 0 and check they match
        # dataframe for all donor records

        donors_list = test_data[test_data["_impute"] == False].index

        for row_index in donors_list:
            rbeis_distances_list = test_data.loc[row_index, "_distances"]
            d1 = RBEISDistanceFunction(
                1, weight=2)(test_data.loc[row_index, self.test_aux_var1],
                             self.igroup0_aux_var1)
            d2 = RBEISDistanceFunction(1, weight=3)(
                test_data.loc[row_index, self.test_aux_var_categorial],
                self.igroup0_aux_categorical,
            )
            weighted_distance = d1 + d2
            self.assertEqual(weighted_distance, rbeis_distances_list[0])

    # --- Test distance calculations using distance function 2  ---
    def test_calc_distances_df2(self):
        test_data = pd.read_csv(self.test_data_filepath)
        _add_impute_col(test_data, self.test_impute_var)
        _assign_igroups(test_data, [self.test_aux_var1, self.test_aux_var2])
        _calc_distances(
            data=test_data,
            aux_vars={
                self.test_aux_var1:
                RBEISDistanceFunction(2,
                                      threshold=self.test_threshold1,
                                      weight=2),
                self.test_aux_var2:
                RBEISDistanceFunction(2,
                                      threshold=self.test_threshold2,
                                      weight=3),
            },
        )

        # Recalculate distances for iGroup 0 and check they match
        # dataframe for all donor records

        donors_list = test_data[test_data["_impute"] == False].index

        for row_index in donors_list:
            rbeis_distances_list = test_data.loc[row_index, "_distances"]
            d1 = RBEISDistanceFunction(2,
                                       threshold=self.test_threshold1,
                                       weight=2)(
                                           test_data.loc[row_index,
                                                         self.test_aux_var1],
                                           self.igroup0_aux_var1)
            d2 = RBEISDistanceFunction(2,
                                       threshold=self.test_threshold2,
                                       weight=3)(
                                           test_data.loc[row_index,
                                                         self.test_aux_var2],
                                           self.igroup0_aux_var2)
            weighted_distance = d1 + d2
            self.assertEqual(weighted_distance, rbeis_distances_list[0])

    # --- Test distance calculations using distance function 3  ---
    def test_calc_distances_df3(self):
        test_data = pd.read_csv(self.test_data_filepath)
        _add_impute_col(test_data, self.test_impute_var)
        _assign_igroups(test_data, [self.test_aux_var1, self.test_aux_var2])
        _calc_distances(
            data=test_data,
            aux_vars={
                self.test_aux_var1:
                RBEISDistanceFunction(3,
                                      threshold=self.test_threshold1,
                                      weight=2),
                self.test_aux_var2:
                RBEISDistanceFunction(3,
                                      threshold=self.test_threshold2,
                                      weight=3),
            },
        )

        # Recalculate distances for iGroup 0 and check they match
        # dataframe for all donor records.

        donors_list = test_data[test_data["_impute"] == False].index

        for row_index in donors_list:
            rbeis_distances_list = test_data.loc[row_index, "_distances"]
            d1 = RBEISDistanceFunction(3,
                                       threshold=self.test_threshold1,
                                       weight=2)(
                                           test_data.loc[row_index,
                                                         self.test_aux_var1],
                                           self.igroup0_aux_var1)
            d2 = RBEISDistanceFunction(3,
                                       threshold=self.test_threshold2,
                                       weight=3)(
                                           test_data.loc[row_index,
                                                         self.test_aux_var2],
                                           self.igroup0_aux_var2)
            weighted_distance = d1 + d2

            # For dfs 3 & 6, assertAlmostEqual is used as calculated distances
            # differ from rbeis distances after about the 7th decimal place
            self.assertAlmostEqual(weighted_distance,
                                   rbeis_distances_list[0],
                                   places=5)

    # --- Test distance calculations using distance function 4  ---
    def test_calc_distances_df4(self):
        test_data = pd.read_csv(self.test_data_filepath)
        _add_impute_col(test_data, self.test_impute_var)
        _assign_igroups(test_data,
                        [self.test_aux_var1, self.test_aux_var_categorial])

        _calc_distances(
            data=test_data,
            aux_vars={
                self.test_aux_var1:
                RBEISDistanceFunction(4,
                                      custom_map=self.test_custom_map1,
                                      weight=2),
                self.test_aux_var_categorial:
                RBEISDistanceFunction(
                    4, custom_map=self.test_custom_map_categorial, weight=3),
            },
        )

        # Recalculate distances for iGroup 0 and check they match
        # dataframe for all donor records

        donors_list = test_data[test_data["_impute"] == False].index

        for row_index in donors_list:
            rbeis_distances_list = test_data.loc[row_index, "_distances"]
            d1 = RBEISDistanceFunction(4,
                                       custom_map=self.test_custom_map1,
                                       weight=2)(
                                           test_data.loc[row_index,
                                                         self.test_aux_var1],
                                           self.igroup0_aux_var1)
            d2 = RBEISDistanceFunction(
                4, custom_map=self.test_custom_map_categorial, weight=3)(
                    test_data.loc[row_index, self.test_aux_var_categorial],
                    self.igroup0_aux_categorical,
                )
            weighted_distance = d1 + d2

            self.assertEqual(weighted_distance, rbeis_distances_list[0])

    # --- Test distance calculations using distance functions 5 & 6 ---
    def test_calc_distances_df5_df6(self):
        test_data = pd.read_csv(self.test_data_filepath)
        _add_impute_col(test_data, self.test_impute_var)
        _assign_igroups(test_data, [self.test_aux_var1, self.test_aux_var2])
        _calc_distances(
            data=test_data,
            aux_vars={
                self.test_aux_var1:
                RBEISDistanceFunction(
                    5,
                    threshold=self.test_threshold1,
                    custom_map=self.test_custom_map1,
                    weight=2,
                ),
                self.test_aux_var2:
                RBEISDistanceFunction(
                    6,
                    threshold=self.test_threshold2,
                    custom_map=self.test_custom_map1,
                    weight=3,
                ),
            },
        )

        # Recalculate distances for iGroup 0 and check they match
        # dataframe for all donor records.

        donors_list = test_data[test_data["_impute"] == False].index

        for row_index in donors_list:
            rbeis_distances_list = test_data.loc[row_index, "_distances"]
            d1 = RBEISDistanceFunction(
                5,
                threshold=self.test_threshold1,
                custom_map=self.test_custom_map1,
                weight=2,
            )(test_data.loc[row_index, self.test_aux_var1],
              self.igroup0_aux_var1)
            d2 = RBEISDistanceFunction(
                6,
                threshold=self.test_threshold2,
                custom_map=self.test_custom_map1,
                weight=3,
            )(test_data.loc[row_index, self.test_aux_var2],
              self.igroup0_aux_var2)
            weighted_distance = d1 + d2

            # For dfs 3 & 6, assertAlmostEqual is used as calculated distances
            # differ from rbeis distances after about the 7th decimal place
            self.assertAlmostEqual(weighted_distance,
                                   rbeis_distances_list[0],
                                   places=5)


# -----------------------------------------------------------------------------
# TESTS: Function: _calc_donors
# This function adds a column 'donor' containing a list of IGroup numbers to
# which each record is a donor.
# -----------------------------------------------------------------------------
class TestCalcDonors(RBEISTestCase):

    # Distance funtion 1 & 3 chosen, with ratio set to None and to 2

    # --- Test list of iGroups each donor can donate to is correct  ---
    def test_calc_donors(self):
        test_data = pd.read_csv(self.test_data_filepath)
        _add_impute_col(test_data, self.test_impute_var)
        _assign_igroups(test_data, [self.test_aux_var1, self.test_aux_var2])
        _calc_distances(
            data=test_data,
            aux_vars={
                self.test_aux_var1: RBEISDistanceFunction(1, weight=2),
                self.test_aux_var2: RBEISDistanceFunction(
                        3,
                        threshold=self.test_threshold2, 
                        weight=3)
            },
        )

        # Test for ratio = None
        _calc_donors(test_data, ratio=None)

        # Check recipients have empty list in donors column
        recipients = test_data[test_data["_impute"] == True]
        for row_index in range(recipients.shape[0]):
            self.assertTrue(len(recipients["_donor"].values[row_index]) == 0)

        # Check donor records have list of correct iGroups
        for igroup_nummber in range(1 + test_data["_IGroup"].max()):
            # Calculate the minumum sum weighted distance for iGroup
            test_data["IGroup_distances"] = [
                dist[igroup_nummber] if len(dist) > 0 else np.nan
                for dist in test_data["_distances"]
            ]
            min_distance = test_data["IGroup_distances"].min()
            # For ratio = None, donors with minimum distance are chosen
            igroup_donors = test_data[test_data["IGroup_distances"] ==
                                      min_distance].index
            for row_index in igroup_donors:                
                self.assertTrue(
                    igroup_nummber in test_data.loc[row_index, "_donor"])
  
       # Test for ratio = 2 
        # Drop _donor column and call _calc_donors to re-add the column for ratio = 2
        test_data = test_data.drop('_donor', axis=1)
        _calc_donors(test_data, ratio=2)
        
        # Check recipients have empty list in donors column
        recipients = test_data[test_data["_impute"] == True]
        for row_index in range(recipients.shape[0]):
            self.assertTrue(len(recipients["_donor"].values[row_index]) == 0)

        # Check donors have list of correct iGroups
        for igroup_nummber in range(1 + test_data["_IGroup"].max()):
            # Calculate the minumum sum weighted distance for iGroup           
            test_data["IGroup_distances"] = [
                dist[igroup_nummber] if len(dist) > 0 else np.nan
                for dist in test_data["_distances"]
            ]          
            min_distance = test_data["IGroup_distances"].min()
            # For ratio = 2, donors within 2 times minimum distance are chosen  
            igroup_donors = test_data[test_data["IGroup_distances"] <=
                                      2*min_distance].index
            for row_index in igroup_donors:                
                self.assertTrue(
                    igroup_nummber in test_data.loc[row_index, "_donor"])
       

# -----------------------------------------------------------------------------
# TESTS: Function: _get_donors
# This function returns a list of indices corresponding to records that
# are donors to the specified IGroup.
# -----------------------------------------------------------------------------
class TestGetDonors(RBEISTestCase):

    # Test list of donors for each iGroup ties up with list of
    # iGroups for each donor.
    # Distance funtions 4 & 5 chosen.

    # --- Test list of donors for each iGroup is correct ---
    def test_get_donors(self):
        test_data = pd.read_csv(self.test_data_filepath)
        _add_impute_col(test_data, self.test_impute_var)
        _assign_igroups(test_data, [self.test_aux_var1, self.test_aux_var2])
        _calc_distances(
            data=test_data,
            aux_vars={
                self.test_aux_var1:
                RBEISDistanceFunction(
                    5,
                    threshold=self.test_threshold1,
                    custom_map=self.test_custom_map1,
                    weight=3),
                self.test_aux_var_categorial:
                RBEISDistanceFunction(
                    4, custom_map=self.test_custom_map_categorial, weight=4),
            },
        )
        _calc_donors(test_data, ratio=None)

        # Test list of donors for each iGroup ties up with list of
        # iGroups for each donor
        for igroup_nummber in range(1 + test_data["_IGroup"].max()):
            donors_for_igroup = _get_donors(test_data, igroup=igroup_nummber)
            row_indices_donors = np.where(donors_for_igroup)[0]
            for row_index in row_indices_donors:
                self.assertTrue(igroup_nummber in test_data.loc[row_index,
                                                                "_donor"])


# -----------------------------------------------------------------------------
# TESTS: Function: _get_freq_dis
# For a given IGroup, this function returns a frequency distribution for each
# possible value of the variable to be imputed. This takes the form of a list
# of the proportions of a given igroup taken up by each possible value.
# -----------------------------------------------------------------------------
class TestGetFrequencyDistribution(RBEISTestCase):

    # Distance funtions 4 & 5 chosen, with ratio set to None and to 3

    # --- Test frequency distribution is calulated correctly for each iGroup --
    def test_get_freq_dist(self):
        test_data = pd.read_csv(self.test_data_filepath)
        _add_impute_col(test_data, self.test_impute_var)
        _assign_igroups(test_data, [self.test_aux_var1, self.test_aux_var2])
        _calc_distances(
            data=test_data,
            aux_vars={
                self.test_aux_var1:
                RBEISDistanceFunction(
                    5,
                    threshold=self.test_threshold1,
                    custom_map=self.test_custom_map1,
                    weight=3,
                ),
                self.test_aux_var_categorial:
                RBEISDistanceFunction(
                    4, custom_map=self.test_custom_map_categorial, weight=4),
            },
        )

        # Test for ratio set to None (default ratio=1)
        _calc_donors(data=test_data, ratio=None)

        # For each iGroup, test that the probabilities assigned to possible
        # values add up to 1
        for igroup_number in range(1 + test_data["_IGroup"].max()):
            freq_dist_list = _get_freq_dist(
                data=test_data,
                imp_var=self.test_impute_var,
                possible_vals=self.test_pos_vals,
                igroup=igroup_number,
            )
            self.assertTrue(sum(freq_dist_list) == Fraction(1, 1))

        # Test for ratio = 3 
        # Drop _donor column and call _calc_donors to re-add the column for ratio = 3
        test_data = test_data.drop('_donor', axis=1)
        _calc_donors(test_data, ratio=3)
        
        # For each iGroup, test that the probabilities assigned to possible
        # values add up to 1
        for igroup_number in range(1 + test_data["_IGroup"].max()):
            freq_dist_list = _get_freq_dist(
                data=test_data,
                imp_var=self.test_impute_var,
                possible_vals=self.test_pos_vals,
                igroup=igroup_number
            )
            self.assertTrue(sum(freq_dist_list) == Fraction(1, 1))
       

# -----------------------------------------------------------------------------
# TESTS: Function: _freq_to_exp
# This function converts a frequency distribution to the expected numbers
# of occurrences for a given IGroup.
# -----------------------------------------------------------------------------
class TestFrequencyToExpected(RBEISTestCase):

    # Distance funtions 4 & 6 chosen

    # --- Test frequency is correctly translated into expected numbers for 
    #     each iGroup ---
    def test_freq_to_exp(self):
        test_data = pd.read_csv(self.test_data_filepath)
        _add_impute_col(test_data, self.test_impute_var)
        _assign_igroups(test_data, [self.test_aux_var1, self.test_aux_var2])
        _calc_distances(
            data=test_data,
            aux_vars={
                self.test_aux_var1:
                RBEISDistanceFunction(
                    6,
                    threshold=self.test_threshold1,
                    custom_map=self.test_custom_map1,
                    weight=3,
                ),
                self.test_aux_var_categorial:
                RBEISDistanceFunction(
                    4, custom_map=self.test_custom_map_categorial, weight=4),
            },
        )
        _calc_donors(data=test_data, ratio=None)

        # Calculate the size of each iGroup
        recipients = test_data[test_data[self.test_impute_var].isnull()]
        igroup_size = recipients.groupby(by="_IGroup")["_IGroup"].count()

        # For each iGroup, test that the expected values assigned to
        # possible values add up to the number in each iGroup

        for igroup_number in range(1 + test_data["_IGroup"].max()):
            freq_dist_list = _get_freq_dist(
                data=test_data,
                imp_var=self.test_impute_var,
                possible_vals=self.test_pos_vals,
                igroup=igroup_number,
            )
            expected_list = _freq_to_exp(test_data, freq_dist_list,
                                         igroup_number)
            self.assertTrue(
                sum(expected_list) == Fraction(igroup_size[igroup_number], 1))


# -----------------------------------------------------------------------------
# TESTS: Function: _impute_igroup
# This function returns a set of imputed values for the given IGroup.
# -----------------------------------------------------------------------------
class TestImputeIGroup(RBEISTestCase):

    # Distance funtiosn 4 & 6 chosen

    # --- Test imputed values returned for each iGroup ---
    def test_impute_igroup(self):

        test_data = pd.read_csv(self.test_data_filepath)
        _add_impute_col(test_data, self.test_impute_var)
        _assign_igroups(test_data, [self.test_aux_var1, self.test_aux_var2])
        _calc_distances(
            data=test_data,
            aux_vars={
                self.test_aux_var1:
                RBEISDistanceFunction(
                    6,
                    threshold=self.test_threshold1,
                    custom_map=self.test_custom_map1,
                    weight=3,
                ),
                self.test_aux_var_categorial:
                RBEISDistanceFunction(
                    4, custom_map=self.test_custom_map_categorial, weight=4),
            },
        )
        _calc_donors(data=test_data, ratio=None)

        # For each iGroup, check that all values in list have expected number
        # occurences > 0 and if expected number of occurences > 1,
        # then value appears the appropriate number of times

        for igroup_number in range(1 + test_data["_IGroup"].max()):
            freq_dist_list = _get_freq_dist(
                data=test_data,
                imp_var=self.test_impute_var,
                possible_vals=self.test_pos_vals,
                igroup=igroup_number,
            )
            expected_list = _freq_to_exp(test_data, freq_dist_list,
                                         igroup_number)

            # save a copy as expected_list will get changed by _impute_igroup
            # when values are imputed
            save_expected_list = expected_list.copy()

            imputed_list = _impute_igroup(
                data=test_data,
                exp_dist=expected_list,
                possible_vals=self.test_pos_vals,
                igroup=igroup_number,
            )

            # Check each imputed value has an expected number ocurences > 0
            for imputed_val in imputed_list:
                index = self.test_pos_vals.index(imputed_val)
                expected_num_occurences = save_expected_list[index]
                self.assertTrue(float(expected_num_occurences) > 0)

            # Check values with expected number ocurences > 1 appear
            # appropriate number of times as imputed values
            for pos_impute_val in self.test_pos_vals:
                index = self.test_pos_vals.index(pos_impute_val)
                expected_num_occurences = float(save_expected_list[index])
                if expected_num_occurences >= 1:
                    num_imputed = imputed_list.count(pos_impute_val)
                    self.assertTrue(
                        (num_imputed == int(expected_num_occurences))
                        or (num_imputed == 1 + int(expected_num_occurences)))


# -----------------------------------------------------------------------------
# TEST: OUTPUT DATAFRAME IS AS EXPECTED
# Check 
# 1. Original dataframe is modified if inplace is TRUE or 
#     unmodified and new datfrane returned if inplace is FALSE  
# 2. Appropriate number of columns depending on whether 
#     keep_intermediatesis TRUE
# -----------------------------------------------------------------------------
class TestOutputDataframe(RBEISTestCase):
        
    # --- Check that 1 column is added when keep_intermediates is FALSE
    def test_keep_intermediates_false(self):

        test_data = pd.read_csv(self.test_data_filepath)
        nrows_orig, ncols_orig = test_data.shape
        impute(
            data=test_data,
            imp_var=self.test_impute_var,
            possible_vals=self.test_pos_vals,
            aux_vars={
                    self.test_aux_var1: 
                        RBEISDistanceFunction(1, weight=2), 
                    self.test_aux_var2: 
                        RBEISDistanceFunction(1, weight=3)
                    },
            ratio=None,
            keep_intermediates=False
        )
        nrows_new, ncols_new = test_data.shape  
        self.assertTrue(ncols_new == 1+ncols_orig) 

    # --- Check that 5 columns are added when keep_intermediates is TRUE  
    def test_keep_intermediates_true(self):

        test_data = pd.read_csv(self.test_data_filepath)
        nrows_orig, ncols_orig = test_data.shape
        impute(
            data=test_data,
            imp_var=self.test_impute_var,
            possible_vals=self.test_pos_vals,
            aux_vars={
                    self.test_aux_var1: 
                        RBEISDistanceFunction(1, weight=2), 
                    self.test_aux_var2: 
                        RBEISDistanceFunction(1, weight=3)
                    },
            ratio=None,
            keep_intermediates=True
        )
        nrows_new, ncols_new = test_data.shape  
        self.assertTrue(ncols_new == 5+ncols_orig) 
        

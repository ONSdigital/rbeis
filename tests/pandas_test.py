from unittest import TestCase

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "../src/rbeis")
sys.path.insert(0, "src/rbeis")

from rbeis import (
    impute,
    _df1,
    _df2,
    _df3,
    _build_custom_df,
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


# Procedures to run before unit tests, if necessary
def setUpModule():

    # --- Set up simple dummy dataframe for testing input parameters for impute function  ---
    dummy_data = {
        "dummy_impute_var": [30, 20, 30, 40, None],
        "dummy_aux_var1": [21, 19, 18, 67, 20],
        "dummy_aux_var2": [2, 3, 1, 1, 3],
        "dummy_aux_var_missing": [4, 3, 6, None, 6],
    }
    dummy_dataframe = pd.DataFrame(dummy_data)

    # --- Set up variables for importing test data ---
    test_data_filepath = "artists_unique_galcount_spaceavg_missing.csv"
    test_impute_var = "whitney_count"
    test_pos_vals = list(range(0, 41))
    test_aux_var1 = "moma_count"
    test_aux_var2 = "space_ratio_per_page_avg"
    test_aux_var_catagorical = "artist_nationality_other"
    test_custom_df_map = {(0, 1): 0.5}

    # Set up data for testing distances calulations
    # For IGroup 0, moma_count = 0,
    # space_ratio_per_page_avg = 0.197291
    # & artist_nationality_other = "American"
    igroup0_aux_var1 = 0
    igroup0_aux_var2 = 0.197291
    igroup0_aux_catagorical = "American"

    pass


# --------------------------------------------------------------------------------------
# TESTS: These tests work with rbeis pandas code version 13-02-2023 before planned changes
# --------------------------------------------------------------------------------------

# ====================================================================================
# --------------- TESTING TEMPLATE ---------------------------
# ====================================================================================
# --- Test type validation on all input parameters ---
# --- Test constrainst on input parameters:
# ---     impute variable and auxiliary variables are in the dataframe ---
# ---     auxiliry variables have no missing values  ---
# ---     weights and distance functions are within range  ---
# ---     threshold and custom_df_map are specified for, and only for, appropriate distance functions  ---
# --- Test each function does what it is expected to do ---
# --- Test if output dataframe is as expected, both new columns and data content ---

# TEST DATA
# Data is loaded for each function test as the order tests are carried out in cannot be controlled

# ====================================================================================

# --------------------------------------------------------------------------------------
# TESTS: IMPUTE MAIN METHOD
# --------------------------------------------------------------------------------------


class TestImpute(TestCase):

    # --------------------------------
    # --- TYPE VALIDATION TESTS    ---
    # --------------------------------

    # --- Test type validation on the input dataframe  ---
    def test_type_validation_data(self):
        with self.assertRaises(TypeError):
            impute(
                data=["Not_a_Dataframe"],
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=1,
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test type validation on the impute variable  ---
    def test_type_validation_imp_var(self):
        with self.assertRaises(TypeError):
            impute(
                data=dummy_dataframe,
                imp_var=["Not", "A", "String"],
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=1,
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test type validation on possible values for impute variable  ---
    def test_type_validation_possible_vals(self):
        with self.assertRaises(TypeError):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals="Not_a_List",
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=1,
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test type validation on auxiliary variables  ---
    def test_type_validation_aux_vars(self):
        with self.assertRaises(TypeError):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=[["Not", "A", "String"], "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=1,
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test type validation on weights  ---
    def test_type_validation_weights(self):
        with self.assertRaises(TypeError):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights="Not_a_Dictionary",
                dist_func=1,
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test type validation on distance function  ---
    def test_type_validation_dist_func(self):
        with self.assertRaises(TypeError):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func="Not_a_Number",
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test type validation on threshhold  ---
    def test_type_validation_threshold(self):
        with self.assertRaises(TypeError):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=2,
                threshold="Not_a_Number",
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test type validation on custom_df_map  ---
    def test_type_validation_custom_df_map(self):
        with self.assertRaises(TypeError):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=4,
                threshold=None,
                custom_df_map="Not_a_Dictionary",
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test type validation on min_quantile  ---
    def test_type_validation_min_quantile(self):
        with self.assertRaises(TypeError):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=1,
                threshold=None,
                custom_df_map=None,
                min_quantile="Not_a_number",
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test type validation on overwrite  ---
    def test_type_validation_overwrite(self):
        with self.assertRaises(TypeError):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=1,
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite="Not_a_Boolean",
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test type validation on col_name  ---
    def test_type_validation_col_name(self):
        with self.assertRaises(TypeError):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=1,
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=["Not", "A", "String"],
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test type validation on in_place  ---
    def test_type_validation_in_place(self):
        with self.assertRaises(TypeError):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=1,
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place="Not_a_Boolean",
                keep_intermediates=True,
            )

    # --- Test type validation on keep_intermediates  ---
    def test_type_validation_keep_intermediates(self):
        with self.assertRaises(TypeError):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=1,
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates="Not_a_Boolean",
            )

    # -----------------------------------
    # --- CHECK INPUT VARIABLES TESTS ---
    # -----------------------------------

    # --- Test exception when impute variable is not in the dataframe  ---
    def test_imp_var_in_df(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="not_a_variable",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=1,
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test exception when possible_vals do not match range of impute variable  ---
    # def test_pos_vals_match_range(self):
    #    with self.assertRaises(Exception):
    #        impute(
    #          data=dummy_dataframe,
    #          imp_var="dummy_impute_var",
    #          possible_vals=list(range(100,120)),
    #          aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
    #          weights={"dummy_aux_var1": 2, "dummy_aux_var2": 3},
    #          dist_func=1,
    #          threshold=None,
    #          custom_df_map=None,
    #          min_quantile=None,
    #          overwrite=False,
    #          col_name=None,
    #          in_place=True,
    #          keep_intermediates=True)

    # --- Test exception when auxiliary variable is not a variable in the dataframe
    def test_aux_vars_in_df(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "not_a_variable"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=1,
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test exception when auxiliary variable has missing values
    def test_aux_vars_no_missing_vals(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var_missing"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var_missing": 3
                },
                dist_func=1,
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test exception when weights dictionary keys do not match auxiliary variables
    def test_weights_keys_match_aux_vars(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "not_a_variable": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=1,
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test exception when weights are not positive values
    def test_weights_are_positive(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": -1
                },
                dist_func=1,
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test exception when dist_func is not an int between 1 and 6
    def test_dist_func_1_to_6(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=7,
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test exception when threshold IS specified for distance function 1
    def test_threshold_for_df1(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=1,
                threshold=3,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test exception when threshold NOT specified for distance function 2
    def test_threshold_for_df2(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=2,
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test exception when threshold NOT specified for distance function 3
    def test_threshold_for_df3(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=3,
                threshold=None,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test exception when threshold IS specified for distance function 4
    def test_threshold_for_df4(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=4,
                threshold=3,
                custom_df_map={(1, 1): 2},
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test exception when threshold NOT specified for distance function 5
    def test_threshold_for_df5(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=5,
                threshold=None,
                custom_df_map={(1, 1): 2},
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test exception when threshold NOT specified for distance function 6
    def test_threshold_for_df6(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=6,
                threshold=None,
                custom_df_map={(1, 1): 2},
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test exception when custom_df_map IS specified for distance function 1
    def test_custom_map_for_df1(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=1,
                threshold=None,
                custom_df_map={(1, 1): 2},
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test exception when custom_df_map IS specified for distance function 2
    def test_custom_map_for_df2(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=2,
                threshold=1,
                custom_df_map={(1, 1): 2},
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test exception when custom_df_map IS specified for distance function 3
    def test_custom_map_for_df3(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=3,
                threshold=1,
                custom_df_map={(1, 1): 2},
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test exception when custom_df_map NOT specified for distance function 4
    def test_custom_map_for_df4(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=4,
                threshold=1,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test exception when custom_df_map NOT specified for distance function 5
    def test_custom_map_for_df5(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=5,
                threshold=1,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )

    # --- Test exception when custom_df_map NOT specified for distance function 6
    def test_custom_map_for_df6(self):
        with self.assertRaises(Exception):
            impute(
                data=dummy_dataframe,
                imp_var="dummy_impute_var",
                possible_vals=list(range(1, 101)),
                aux_vars=["dummy_aux_var1", "dummy_aux_var2"],
                weights={
                    "dummy_aux_var1": 2,
                    "dummy_aux_var2": 3
                },
                dist_func=6,
                threshold=1,
                custom_df_map=None,
                min_quantile=None,
                overwrite=False,
                col_name=None,
                in_place=True,
                keep_intermediates=True,
            )


# --------------------------------------------------------------------------------------
# TESTS: DISTANCE FUNCTIONS: _df1, _df2, _df3, _build_custom_df
#
# It is assumed that missing values cannot be passed to the distance functions
# as imputation will be prevented when the impute function is called
# if any of the auxiliary values contain missing values
# --------------------------------------------------------------------------------------


class TestDistanceFunctions(TestCase):

    # --- Test distance function 1 returns correct values ---
    def test_df1(self):
        self.assertEqual(_df1(1, 1, None), 0)
        self.assertEqual(_df1("abcd", "abcd", None), 0)
        self.assertEqual(_df1(0, 1, None), 1)
        self.assertEqual(_df1(2.3, 1.2, 20), 1)
        self.assertEqual(_df1("abcd", "bcde", None), 1)

    # --- Test distance function 2 returns correct values ---
    def test_df2(self):
        self.assertEqual(_df2(1, 1, 3), 0)
        self.assertEqual(_df2(1.1, 3.5, 3.0), 0)
        self.assertEqual(_df2(1, 5, 3), 1)
        self.assertEqual(_df2(7.8, 2.1, 3.5), 1)

    # --- Test distance function 3 returns correct values ---
    def test_df3(self):
        self.assertEqual(_df3(1, 1, 3), 0)
        self.assertEqual(_df3(1, 3, 3), 0.5)
        self.assertEqual(_df3(7, 6, 3), 0.25)
        self.assertEqual(_df3(7, 2, 3), 1)

    # Test Exception is raised for df values other than 4, 5 or 6
    def test_build_custom_df(self):
        with self.assertRaises(Exception):
            rbeis._build_custom_df(7, {(1, 1): 2})

    # --- Test distance function 4 returns correct values ---
    def test_df4(self):
        df4 = _build_custom_df(4, {(1, 1): 2, (2.2, 3.3): 8})
        self.assertEqual(df4(2.2, 3.3, None), 8.0)
        self.assertEqual(df4(2, 3, None), 1)
        self.assertEqual(df4(2, 2, None), 0)
        self.assertEqual(df4("abcd", "abcd", None), 0)
        self.assertEqual(df4(1.1, 2.2, 1), 1)
        self.assertEqual(df4("abcd", "bcde", None), 1)

    # --- Test distance function 5 returns correct values ---
    def test_df5(self):
        df5 = _build_custom_df(5, {(1, 1): 2, (2, 3): 8}, 3)
        self.assertEqual(df5(1, 1, 3), 2)
        self.assertEqual(df5(1, 4, 3), 0)
        self.assertEqual(df5(1.1, 5.5, 3), 1)
        self.assertEqual(df5(7, 2, 3), 1)

    # --- Test distance function 6 returns correct values ---
    def test_df6(self):
        df6 = _build_custom_df(6, {(1, 1): 2}, 3.0)
        self.assertEqual(df6(1, 1, 3), 2.0)
        self.assertEqual(df6(5, 5, 3), 0)
        self.assertEqual(df6(1, 3, 3), 0.5)
        self.assertEqual(df6(7, 6, 3), 0.25)
        self.assertEqual(df6(7, 2, 3), 1)


# -----------------------------------------------------------------------------
# TESTS: Function: _add_impute_col
# This function adds a boolean column 'impute' indicating whether record
# will be imputed.
# -----------------------------------------------------------------------------


# --- Test impute column values are assigned correctly ---
class TestAddImputeCol(TestCase):

    def test_assign_impute(self):
        test_data = pd.read_csv(test_data_filepath)
        _add_impute_col(test_data, test_impute_var)

        # Check that correct Boolean values are assigned for _impute column
        self.assertTrue(test_data["_impute"].equals(
            np.isnan(test_data[test_impute_var])))


# -----------------------------------------------------------------------------
# TESTS: Function: _assign_igroups
# This function adds a column 'IGroup' containing integers representing
# the IGroup each recipient record is assigned to.
# -----------------------------------------------------------------------------


# --- Test Igroup column values are assigned correctly ---
class TestAssignIgroups(TestCase):

    def test_assign_igroups(self):
        test_data = pd.read_csv(test_data_filepath)
        _add_impute_col(test_data, test_impute_var)
        _assign_igroups(test_data, [test_aux_var1, test_aux_var2])

        # Check Igroup is set to -1 for non-recipient records
        donors = test_data[test_data["_impute"] == False]
        for row_index in range(donors.shape[0]):
            self.assertTrue(donors["_IGroup"].values[row_index] == -1)

        # Check Igroup isassigned for recipiants
        recipiants = test_data[test_data["_impute"] == True]
        for row_index in range(recipiants.shape[0]):
            self.assertTrue(recipiants["_IGroup"].values[row_index] > -1)

        # Calculate how many recipiants in each IGroup
        recipiant_freq = recipiants.groupby(by="_IGroup")["_IGroup"].count()

        # Check records assigned to same IGroup have same valueas for
        # auxiliary variables
        multiple_recipiant_igroups = recipiant_freq[recipiant_freq > 1].index
        for igroup_number in multiple_recipiant_igroups:
            igroup_data = test_data[test_data["_IGroup"] == igroup_number]
            self.assertTrue(len(igroup_data[test_aux_var1].unique()) == 1)
            self.assertTrue(len(igroup_data[test_aux_var2].unique()) == 1)

        # Check single recipiant IGroups all have different combination of
        # auxiliary variables
        single_recipiant_list = recipiant_freq[recipiant_freq == 1].index
        single_recipiant_data = test_data.copy()
        single_recipiant_data = single_recipiant_data[
            single_recipiant_data["_IGroup"].isin(single_recipiant_list)]

        single_recipiant_data["combine_aux_vars"] = (
            single_recipiant_data[test_aux_var1].astype(str) + " " +
            single_recipiant_data[test_aux_var2].astype(str))

        single_recipiant_freq = single_recipiant_data.groupby(
            by="combine_aux_vars")["combine_aux_vars"].count()

        self.assertTrue(
            len(single_recipiant_freq[single_recipiant_freq > 1].index) == 0)


# -----------------------------------------------------------------------------
# TESTS: Function: _get_igroup_aux_var
# This function returns a list containing each IGroup's value of a given
# auxiliary variable.
# -----------------------------------------------------------------------------


# --- Test list of auxiliary variables values for each iGroup is correct ---
class TestGetIGroupAuxVar(TestCase):

    def test_get_igroup_aux_var(self):
        test_data = pd.read_csv(test_data_filepath)
        _add_impute_col(test_data, test_impute_var)
        _assign_igroups(test_data, [test_aux_var1, test_aux_var2])

        # Get list of auxiliary variable values for each iGroup
        aux_var1_list = _get_igroup_aux_var(test_data, test_aux_var1)
        aux_var2_list = _get_igroup_aux_var(test_data, test_aux_var2)

        # Check records assigned to each iGroup match auxiliary values in list
        for igroup_number in range(1 + test_data["_IGroup"].max()):
            for row_index in range(test_data.shape[0]):
                if test_data["_IGroup"].values[row_index] == igroup_number:
                    self.assertTrue(test_data[test_aux_var1].values[row_index]
                                    == aux_var1_list[igroup_number])
                    self.assertTrue(test_data[test_aux_var2].values[row_index]
                                    == aux_var2_list[igroup_number])


# -----------------------------------------------------------------------------
# TESTS: Function: _calc_distances
# This function adds a column '_distances' containing lists of calculated
# distances of each record's auxiliary variables from those of its IGroup.
# -----------------------------------------------------------------------------


# --- Test distances are calulated correctly for each distance function ---
# --- for iGroup 0 only ---
class TestCalcDistances(TestCase):

    # --- Test distance calculations using distance function 1  ---
    def test_calc_distances_df1(self):
        test_data = pd.read_csv(test_data_filepath)
        _add_impute_col(test_data, test_impute_var)
        _assign_igroups(test_data, [test_aux_var1, test_aux_var_catagorical])
        _calc_distances(
            data=test_data,
            aux_vars=[test_aux_var1, test_aux_var_catagorical],
            dist_func=1,
            weights={
                test_aux_var1: 2,
                test_aux_var_catagorical: 3
            },
            threshold=None,
            custom_df_map=None,
        )

        # Recalculate distances for iGroup 0 and check they match
        # dataframe for all donor records

        donors_list = test_data[test_data["_impute"] == False].index

        for row_index in donors_list:
            rbeis_distances_list = test_data.loc[row_index, "_distances"]
            d1 = _df1(igroup0_aux_var1, test_data.loc[row_index,
                                                      test_aux_var1], None)
            d2 = _df1(
                igroup0_aux_catagorical,
                test_data.loc[row_index, test_aux_var_catagorical],
                None,
            )
            weighted_distance = 2 * d1 + 3 * d2
            self.assertEqual(weighted_distance, rbeis_distances_list[0])

    # --- Test distance calculations using distance function 2  ---
    def test_calc_distances_df2(self):
        test_data = pd.read_csv(test_data_filepath)
        _add_impute_col(test_data, test_impute_var)
        _assign_igroups(test_data, [test_aux_var1, test_aux_var2])
        _calc_distances(
            data=test_data,
            aux_vars=[test_aux_var1, test_aux_var2],
            dist_func=2,
            weights={
                test_aux_var1: 2,
                test_aux_var2: 3
            },
            threshold=0.1,
            custom_df_map=None,
        )

        # Recalculate distances for iGroup 0 and check they match
        # dataframe for all donor records

        donors_list = test_data[test_data["_impute"] == False].index

        for row_index in donors_list:
            rbeis_distances_list = test_data.loc[row_index, "_distances"]
            d1 = _df2(igroup0_aux_var1, test_data.loc[row_index,
                                                      test_aux_var1], 0.1)
            d2 = _df2(igroup0_aux_var2, test_data.loc[row_index,
                                                      test_aux_var2], 0.1)
            weighted_distance = 2 * d1 + 3 * d2
            self.assertEqual(weighted_distance, rbeis_distances_list[0])

    # --- Test distance calculations using distance function 3  ---
    def test_calc_distances_df3(self):
        test_data = pd.read_csv(test_data_filepath)
        _add_impute_col(test_data, test_impute_var)
        _assign_igroups(test_data, [test_aux_var1, test_aux_var2])
        _calc_distances(
            data=test_data,
            aux_vars=[test_aux_var1, test_aux_var2],
            dist_func=3,
            weights={
                test_aux_var1: 2,
                test_aux_var2: 3
            },
            threshold=0.1,
            custom_df_map=None,
        )

        # Recalculate distances for iGroup 0 and check they match
        # dataframe for all donor records.
        # For dfs 3 & 6, assertAlmostEqual is used as calculated distances
        # differ from rbeis distances after about the 7th the decimal place

        donors_list = test_data[test_data["_impute"] == False].index

        for row_index in donors_list:
            rbeis_distances_list = test_data.loc[row_index, "_distances"]
            d1 = _df3(igroup0_aux_var1, test_data.loc[row_index,
                                                      test_aux_var1], 0.1)
            d2 = _df3(igroup0_aux_var2, test_data.loc[row_index,
                                                      test_aux_var2], 0.1)
            weighted_distance = 2 * d1 + 3 * d2
            self.assertAlmostEqual(weighted_distance,
                                   rbeis_distances_list[0],
                                   places=5)

    # --- Test distance calculations using distance function 4  ---
    def test_calc_distances_df4(self):
        test_data = pd.read_csv(test_data_filepath)
        _add_impute_col(test_data, test_impute_var)
        _assign_igroups(test_data, [test_aux_var1, test_aux_var_catagorical])
        _calc_distances(
            data=test_data,
            aux_vars=[test_aux_var1, test_aux_var_catagorical],
            dist_func=4,
            weights={
                test_aux_var1: 2,
                test_aux_var_catagorical: 3
            },
            threshold=None,
            custom_df_map={
                (0, 1): 0.5,
                ("American", "British"): 0.5
            },
        )

        # Recalculate distances for iGroup 0 and check they match
        # dataframe for all donor records

        donors_list = test_data[test_data["_impute"] == False].index
        df4 = _build_custom_df(4, {
            (0, 1): 0.5,
            ("American", "British"): 0.5
        }, None)

        for row_index in donors_list:
            rbeis_distances_list = test_data.loc[row_index, "_distances"]
            d1 = df4(igroup0_aux_var1, test_data.loc[row_index, test_aux_var1],
                     None)
            d2 = df4(
                igroup0_aux_catagorical,
                test_data.loc[row_index, test_aux_var_catagorical],
                None,
            )
            weighted_distance = 2 * d1 + 3 * d2
            self.assertEqual(weighted_distance, rbeis_distances_list[0])

    # --- Test distance calculations using distance function 5  ---
    def test_calc_distances_df5(self):
        test_data = pd.read_csv(test_data_filepath)
        _add_impute_col(test_data, test_impute_var)
        _assign_igroups(test_data, [test_aux_var1, test_aux_var2])
        _calc_distances(
            data=test_data,
            aux_vars=[test_aux_var1, test_aux_var2],
            dist_func=5,
            weights={
                test_aux_var1: 2,
                test_aux_var2: 3
            },
            threshold=0.1,
            custom_df_map={(0, 1): 0.5},
        )

        # Recalculate distances for iGroup 0 and check they match
        # dataframe for all donor records
        donors_list = test_data[test_data["_impute"] == False].index
        df5 = _build_custom_df(5, {(0, 1): 0.5}, 0.1)

        for row_index in donors_list:
            rbeis_distances_list = test_data.loc[row_index, "_distances"]
            d1 = df5(igroup0_aux_var1, test_data.loc[row_index, test_aux_var1],
                     0.1)
            d2 = df5(igroup0_aux_var2, test_data.loc[row_index, test_aux_var2],
                     0.1)
            weighted_distance = 2 * d1 + 3 * d2
            self.assertEqual(weighted_distance, rbeis_distances_list[0])

    # --- Test distance calculations using distance function 6  ---
    def test_calc_distances_df6(self):
        test_data = pd.read_csv(test_data_filepath)
        _add_impute_col(test_data, test_impute_var)
        _assign_igroups(test_data, [test_aux_var1, test_aux_var2])
        _calc_distances(
            data=test_data,
            aux_vars=[test_aux_var1, test_aux_var2],
            dist_func=6,
            weights={
                test_aux_var1: 2,
                test_aux_var2: 3
            },
            threshold=0.1,
            custom_df_map={(0, 1): 0.5},
        )

        # Recalculate distances for iGroup 0 and check they match dataframe
        # for all donor records.
        # For dfs 3 & 6, assertAlmostEqual is used as calculated distances
        # differ from rbeis distances after about the 7th the decimal place

        donors_list = test_data[test_data["_impute"] == False].index
        df6 = _build_custom_df(6, {(0, 1): 0.5}, 0.1)

        for row_index in donors_list:
            rbeis_distances_list = test_data.loc[row_index, "_distances"]
            d1 = df6(igroup0_aux_var1, test_data.loc[row_index, test_aux_var1],
                     0.1)
            d2 = df6(igroup0_aux_var2, test_data.loc[row_index, test_aux_var2],
                     0.1)
            weighted_distance = 2 * d1 + 3 * d2
            self.assertAlmostEqual(weighted_distance,
                                   rbeis_distances_list[0],
                                   places=5)


# -----------------------------------------------------------------------------
# TESTS: Function: _calc_donors
# This function adds a column 'donor' containing a list of IGroup numbers to
# which each record is a donor.
# -----------------------------------------------------------------------------


# --- Test list of iGroups each donor can donate to is correct  ---
class TestCalcDonors(TestCase):

    # Test for min_quantile = None as min_quantile will be changed to ratio
    # Distance funtion 1 chosen

    def test_calc_donors(self):
        test_data = pd.read_csv(test_data_filepath)
        _add_impute_col(test_data, test_impute_var)
        _assign_igroups(test_data, [test_aux_var1, test_aux_var2])
        _calc_distances(
            data=test_data,
            aux_vars=[test_aux_var1, test_aux_var2],
            dist_func=1,
            weights={
                test_aux_var1: 2,
                test_aux_var2: 3
            },
        )
        _calc_donors(test_data, min_quantile=None)

        # Check recipiants have empty list in donors column
        recipiants = test_data[test_data["_impute"] == True]
        for row_index in range(recipiants.shape[0]):
            self.assertTrue(len(recipiants["_donor"].values[row_index]) == 0)

        # Check donors have list of correct iGroups
        for igroup_nummber in range(1 + test_data["_IGroup"].max()):
            test_data["IGroup_distances"] = [
                dist[igroup_nummber] if len(dist) > 0 else np.nan
                for dist in test_data["_distances"]
            ]
            min_distance = test_data["IGroup_distances"].min()
            igroup_donors = test_data[test_data["IGroup_distances"] ==
                                      min_distance].index
            for row_index in igroup_donors:
                self.assertTrue(igroup_nummber in test_data.loc[row_index,
                                                                "_donor"])


# -----------------------------------------------------------------------------
# TESTS: Function: _get_donors
# This function returns a list of indices corresponding to records that
# are donors to the specified IGroup.
# -----------------------------------------------------------------------------


# --- Test list of donors for each iGroup is correct ---
class TestGetDonors(TestCase):

    # Test list of donors for each iGroup ties up with list of
    # iGroups for each donor
    def test_get_donors(self):

        # Test for min_quantile = None as min_quantile will be changed to ratio
        # Distance funtion 6 chosen

        test_data = pd.read_csv(test_data_filepath)
        _add_impute_col(test_data, test_impute_var)
        _assign_igroups(test_data, [test_aux_var1, test_aux_var2])
        _calc_distances(
            data=test_data,
            aux_vars=[test_aux_var1, test_aux_var2],
            dist_func=6,
            weights={
                test_aux_var1: 3,
                test_aux_var2: 4
            },
            threshold=2,
            custom_df_map=test_custom_df_map,
        )
        _calc_donors(test_data, min_quantile=None)

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


# --- Test frequency distribution is calulated correctly for each iGroup  ---
class TestGetFrequencyDistribution(TestCase):

    # Test for min_quantile = None as min_quantile will be changed to ratio
    # Distance funtion 5 chosen, with threshold of 3

    def test_get_freq_dist(self):
        test_data = pd.read_csv(test_data_filepath)
        _add_impute_col(data=test_data, imp_var=test_impute_var)
        _assign_igroups(data=test_data,
                        aux_vars=[test_aux_var1, test_aux_var2])
        _calc_distances(
            data=test_data,
            aux_vars=[test_aux_var1, test_aux_var2],
            dist_func=5,
            weights={
                test_aux_var1: 2,
                test_aux_var2: 3
            },
            threshold=3,
            custom_df_map=test_custom_df_map,
        )
        _calc_donors(data=test_data, min_quantile=None)

        # For each iGroup, test that the probabilities assigned to possible
        # values add up to 1
        for igroup_number in range(1 + test_data["_IGroup"].max()):
            freq_dist_list = _get_freq_dist(
                data=test_data,
                imp_var=test_impute_var,
                possible_vals=test_pos_vals,
                igroup=igroup_number,
            )
            self.assertTrue(sum(freq_dist_list) == Fraction(1, 1))


# -----------------------------------------------------------------------------
# TESTS: Function: _freq_to_exp
# This function converts a frequency distribution to the expected numbers
# of occurrences for a given IGroup.
# -----------------------------------------------------------------------------


# --- Test frequency is translated in to expected numbers correctly for each iGroup ---
class TestFrequencyToExpected(TestCase):

    # Test for min_quantile = None as min_quantile will be changed to ratio
    # Distance funtion 6 chosen, with threshold of 2

    def test_freq_to_exp(self):
        test_data = pd.read_csv(test_data_filepath)
        _add_impute_col(data=test_data, imp_var=test_impute_var)
        _assign_igroups(data=test_data,
                        aux_vars=[test_aux_var1, test_aux_var2])
        _calc_distances(
            data=test_data,
            aux_vars=[test_aux_var1, test_aux_var2],
            dist_func=6,
            weights={
                test_aux_var1: 2,
                test_aux_var2: 3
            },
            threshold=2,
            custom_df_map=test_custom_df_map,
        )
        _calc_donors(data=test_data, min_quantile=None)

        # Calculate the size of each iGroup
        recipiants = test_data[test_data[test_impute_var].isnull()]
        igroup_size = recipiants.groupby(by="_IGroup")["_IGroup"].count()

        # For each iGroup, test that the expected values assigned to
        # possible values add up to the number in each iGroup

        for igroup_number in range(1 + test_data["_IGroup"].max()):
            freq_dist_list = _get_freq_dist(
                data=test_data,
                imp_var=test_impute_var,
                possible_vals=test_pos_vals,
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


# --- Test imputed values returned for each iGroup ---
class TestImputeIGroup(TestCase):

    # Test for min_quantile = None as min_quantile will be changed to ratio
    # Distance funtion 5 chosen, with threshold of 3

    def test_impute_igroup(self):

        test_data = pd.read_csv(test_data_filepath)
        _add_impute_col(data=test_data, imp_var=test_impute_var)
        _assign_igroups(data=test_data,
                        aux_vars=[test_aux_var1, test_aux_var2])
        _calc_distances(
            data=test_data,
            aux_vars=[test_aux_var1, test_aux_var2],
            dist_func=5,
            weights={
                test_aux_var1: 2,
                test_aux_var2: 3
            },
            threshold=3,
            custom_df_map=test_custom_df_map,
        )
        _calc_donors(data=test_data, min_quantile=None)

        # For each iGroup, check that all values in list have expected number
        # occurences > 0 and if expected number of occurences > 1,
        # then value appears the appropriate number of times

        for igroup_number in range(1 + test_data["_IGroup"].max()):
            freq_dist_list = _get_freq_dist(
                data=test_data,
                imp_var=test_impute_var,
                possible_vals=test_pos_vals,
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
                possible_vals=test_pos_vals,
                igroup=igroup_number,
            )

            # Check each imputed value has an expected number ocurences > 0
            for imputed_val in imputed_list:
                index = test_pos_vals.index(imputed_val)
                expected_num_occurences = save_expected_list[index]
                self.assertTrue(float(expected_num_occurences) > 0)

            # Check values with expected number ocurences > 1 appear
            # appropriate number of times as imputed values
            for pos_impute_val in test_pos_vals:
                index = test_pos_vals.index(pos_impute_val)
                expected_num_occurences = float(save_expected_list[index])
                if expected_num_occurences >= 1:
                    num_imputed = imputed_list.count(pos_impute_val)
                    self.assertTrue(
                        (num_imputed == int(expected_num_occurences))
                        or (num_imputed == 1 + int(expected_num_occurences)))


# Procedures to run after unit tests, if necessary
def tearDownModule():
    pass

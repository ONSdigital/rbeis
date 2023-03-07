from unittest import TestCase

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "../src/rbeis")
sys.path.insert(0, "src/rbeis")

from rbeis_pandas import impute, _df1, _df2, _df3, _build_custom_df, _add_impute_col


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
    test_aux_var1 = "moma_count"
    test_aux_var2 = "space_ratio_per_page_avg"

    pass


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

# CHANGES IN TO BE MADE RBEIS WHICH WILL IMPACT TESTING
# 1. Distance function, threshold and custom_df_map to be set for each auxiliary variable
#    Will need to rewrite impute statements throughout TestImpute class
#    Will need to check that ONLY df1 or df4 can be specified for any categorical aux vars
# 2. May allow impuataion of non-numeric variable if this can be done in python
#    If this cannot be done, then add check that impute variable is numeric
# 3. The min-quantile functionality will be changed back to ratio as in SAS
#    Tests will need to br addedd to check ratio functionality
#    Currently, type validation is carried out on min_quantile and parameter is set to None throughout
# 4. Overwrite parameter functionality may be implemented and will need to be tested
# 5. In-place parameter functionality may be implemented and will need to be tested
# 6. Still to be decided how to check if user specified appropriate range for possible_vals
#    Test will need to be implemented when this is established

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


# --------------------------------------------------------------------------------------
# TESTS: Function: _add_impute_col
# --------------------------------------------------------------------------------------


# --- Test impute column values are assigned correctly ---
class TestAddImputeCol(TestCase):

    def test_assign_impute(self):
        test_data = pd.read_csv(test_data_filepath)
        _add_impute_col(test_data, test_impute_var)

        # Check that correct Boolean values are assigned for _impute column
        self.assertTrue(test_data["_impute"].equals(
            np.isnan(test_data[test_impute_var])))


# Procedures to run after unit tests, if necessary
def tearDownModule():
    pass

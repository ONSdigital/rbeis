from unittest import TestCase

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, '../src/rbeis')
sys.path.insert(0, 'src/rbeis')

from rbeis import (impute, _df1, _df2, _df3, _build_custom_df)

# Procedures to run before unit tests, if necessary
def setUpModule():

    # --- Setup dummy dataframe for testing input parameters ---
    dummy_data = {'age': [30,20,30,40,None],
        'interim_id': [21, 19, 18, 67, 20],
        'work_status_group': [2,3,1,1,3]}
    dummy_dataframe = pd.DataFrame(dummy_data)
    pass

# --------------------------------------------------------------------------------------
# TESTS: IMPUTE MAIN METHOD
# --------------------------------------------------------------------------------------

class TestImpute(TestCase):
    
    # --- TYPE VALIDATION TESTS ---
    
    # --- Test type validation on the input dataframe  ---
    def test_type_validation_data(self):
        with self.assertRaises(TypeError):
            impute(
              data=["Not_a_Dataframe"], 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=1,
              threshold=None, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)
          
    # --- Test type validation on the impute variable  ---
    def test_type_validation_imp_var(self):  
        with self.assertRaises(TypeError):
            impute(
              data=dummy_dataframe, 
              imp_var=["Not","A","String"], 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=1,
              threshold=None, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)
               
    # --- Test type validation on possible values for impute variable  ---
    def test_type_validation_possible_vals(self):         
        with self.assertRaises(TypeError):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals="Not_a_List", 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=1,
              threshold=None, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)
          
    # --- Test type validation on auxiliary variables  ---
    def test_type_validation_aux_vars(self): 
        with self.assertRaises(TypeError):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=[["Not","A","String"], "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=1,
              threshold=None, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)
          
    # --- Test type validation on weights  ---
    def test_type_validation_weights(self):         
        with self.assertRaises(TypeError):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights="Not_a_Dictionary", 
              dist_func=1,
              threshold=None, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)
          
    # --- Test type validation on distance function  ---
    def test_type_validation_dist_func(self):        
        with self.assertRaises(TypeError):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func="Not_a_Number",
              threshold=None, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)
          
    # --- Test type validation on threshhold  ---
    def test_type_validation_threshold(self): 
        with self.assertRaises(TypeError):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=2,
              threshold="Not_a_Number", 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)
          
    # --- Test type validation on custom_df_map  ---
    def test_type_validation_custom_df_map(self): 
        with self.assertRaises(TypeError):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=4,
              threshold=None, 
              custom_df_map="Not_a_Dictionary", 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)
          
    # --- Test type validation on min_quantile  ---
    def test_type_validation_min_quantile(self):       
        with self.assertRaises(TypeError):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=1,
              threshold=None, 
              custom_df_map=None, 
              min_quantile="Not_a_number",
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)   
          
    # --- Test type validation on overwrite  ---
    def test_type_validation_overwrite(self):        
        with self.assertRaises(TypeError):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=1,
              threshold=None, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite="Not_a_Boolean", 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)   
          
    # --- Test type validation on col_name  ---
    def test_type_validation_col_name(self):  
        with self.assertRaises(TypeError):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=1,
              threshold=None, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=["Not","A","String"], 
              in_place=True,
              keep_intermediates=True) 
          
    # --- Test type validation on in_place  ---
    def test_type_validation_in_place(self): 
        with self.assertRaises(TypeError):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=1,
              threshold=None, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place="Not_a_Boolean",
              keep_intermediates=True) 
          
    # --- Test type validation on keep_intermediates  ---
    def test_type_validation_keep_intermediates(self):  
        with self.assertRaises(TypeError):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=1,
              threshold=None, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates="Not_a_Boolean") 
          
    
    # --- CHECK INPUT VARIABLES TESTS ---
              
    # --- Test exception when impute variable is not in the dataframe  ---
    def test_imp_var_in_df(self):
        with self.assertRaises(Exception):
            impute(
              data=dummy_dataframe, 
              imp_var="not_a_variable", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=1,
              threshold=None, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)
          
    # --- Test exception when possible_vals do not match range of impute variable  ---    
    #def test_pos_vals_match_range(self):
    #    with self.assertRaises(Exception):
    #        impute(
    #          data=dummy_dataframe, 
    #          imp_var="age", 
    #          possible_vals=list(range(100,120)), 
    #          aux_vars=["interim_id", "work_status_group"], 
    #          weights={"interim_id": 2, "work_status_group": 3}, 
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
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "not_a_variable"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=1,
              threshold=None, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)
          
    # --- Test exception when weights dictionary keys do not match auxiliary variables
    def test_weights_keys_match_aux_vars(self):  
        with self.assertRaises(Exception):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"not_a_variable": 2, "work_status_group": 3}, 
              dist_func=1,
              threshold=None, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)

    # --- Test exception when dist_func is not an int between 1 and 6
    def test_dist_func_1_to_6(self):  
        with self.assertRaises(Exception):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=7,
              threshold=None, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)

    # --- Test exception when threshold not specified for distance function 2
    def test_threshold_for_df2(self): 
        with self.assertRaises(Exception):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=2,
              threshold=None, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)
          
        # --- Test exception when threshold not specified for distance function 3
    def test_threshold_for_df3(self):         
        with self.assertRaises(Exception):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=3,
              threshold=None, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)
          
        # --- Test exception when threshold not specified for distance function 5
    def test_threshold_for_df5(self): 
        with self.assertRaises(Exception):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=5,
              threshold=None, 
              custom_df_map={(1, 1): 2}, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)
          
    # --- Test exception when threshold not specified for distance function 6
    def test_threshold_for_df6(self): 
        with self.assertRaises(Exception):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=6,
              threshold=None, 
              custom_df_map={(1, 1): 2}, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)
          
    # --- Test exception when custom_df_map not specified for distance function 4
    def test_custom_map_for_df4(self): 
        with self.assertRaises(Exception):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=4,
              threshold=1, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)
          
    # --- Test exception when custom_df_map not specified for distance function 5
    def test_custom_map_for_df5(self):  
        with self.assertRaises(Exception):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=5,
              threshold=1, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)
          
    # --- Test exception when custom_df_map not specified for distance function 6
    def test_custom_map_for_df6(self): 
        with self.assertRaises(Exception):
            impute(
              data=dummy_dataframe, 
              imp_var="age", 
              possible_vals=list(range(1,101)), 
              aux_vars=["interim_id", "work_status_group"], 
              weights={"interim_id": 2, "work_status_group": 3}, 
              dist_func=6,
              threshold=1, 
              custom_df_map=None, 
              min_quantile=None,
              overwrite=False, 
              col_name=None, 
              in_place=True,
              keep_intermediates=True)


# --------------------------------------------------------------------------------------
# TESTS: DISTANCE FUNCTIONS - _df1, _df2, _df3, _build_custom_df
# --------------------------------------------------------------------------------------

class TestDistanceFunctions(TestCase):

    # --- Test distance function 1 returns correct values ---   
    def test_df1(self): 
        self.assertEqual(_df1(1,1,None),0)
        self.assertEqual(_df1('abcd','abcd',None),0)
        self.assertEqual(_df1(0,1,None),1)
        self.assertEqual(_df1(2.3,1.2,20),1)
        self.assertEqual(_df1('abcd','bcde',None),1)

    # --- Test distance function 2 returns correct values ---        
    def test_df2(self): 
        self.assertEqual(_df2(1,1,3),0)
        self.assertEqual(_df2(1.1,3.5,3.0),0)    
        self.assertEqual(_df2(1,5,3),1)
        self.assertEqual(_df2(7.8,2.1,3.5),1)

    # --- Test distance function 3 returns correct values ---   
    def test_df3(self): 
        self.assertEqual(_df3(1,1,3),0)
        self.assertEqual(_df3(1,3,3),0.5)
        self.assertEqual(_df3(7,6,3),0.25)   
        self.assertEqual(_df3(7,2,3),1)

    # Test Exception is raised for df values other than 4, 5 or 6 
    def test_build_custom_df(self): 
      with self.assertRaises(Exception):
          rbeis._build_custom_df(7,{(1,1):2})

    # --- Test distance function 4 returns correct values --- 
    def test_df4(self): 
        df4 = _build_custom_df(4,{(1,1):2,(2.2,3.3):8})
        self.assertEqual(df4(2.2,3.3,None),8.0)
        self.assertEqual(df4(2,3,None),1)
        self.assertEqual(df4(2,2,None),0)
        self.assertEqual(df4('abcd','abcd',None),0)
        self.assertEqual(df4(1.1,2.2,1),1)
        self.assertEqual(df4('abcd','bcde',None),1)

    # --- Test distance function 5 returns correct values ---      
    def test_df5(self): 
        df5 = _build_custom_df(5,{(1,1):2,(2,3):8},3)
        self.assertEqual(df5(1,1,3),2)
        self.assertEqual(df5(1,4,3),0)    
        self.assertEqual(df5(1.1,5.5,3),1)
        self.assertEqual(df5(7,2,3),1)

    # --- Test distance function 6 returns correct values ---      
    def test_df6(self): 
        df6 = _build_custom_df(6,{(1,1):2},3.0)
        self.assertEqual(df6(1,1,3),2.0)
        self.assertEqual(df6(5,5,3),0)
        self.assertEqual(df6(1,3,3),0.5)
        self.assertEqual(df6(7,6,3),0.25)   
        self.assertEqual(df6(7,2,3),1)


# Procedures to run after unit tests, if necessary
def tearDownModule():
    pass

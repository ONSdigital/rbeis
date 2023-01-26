from unittest import TestCase

import numpy as np
import pandas as pd

# Procedures to run before unit tests, if necessary
def setUpModule():
    # TODO: Load a dummy dataset
    pass

class TestDistanceFunctions(TestCase):

  def test_df1(self): 
     self.assertEqual(rbeis._df1(1,1,None),0)
     self.assertEqual(rbeis._df1('abcd','abcd',None),0)
     self.assertEqual(rbeis._df1(0,1,None),1)
     self.assertEqual(rbeis._df1(2.3,1.2,20),1)
     self.assertEqual(rbeis._df1('abcd','bcde',None),1)
     
  def test_df2(self): 
     self.assertEqual(rbeis._df2(1,1,3),0)
     self.assertEqual(rbeis._df2(1.1,3.5,3.0),0)    
     self.assertEqual(rbeis._df2(1,5,3),1)
     self.assertEqual(rbeis._df2(7.8,2.1,3.5),1)

  def test_df3(self): 
     self.assertEqual(rbeis._df3(1,1,3),0)
     self.assertEqual(rbeis._df3(1,3,3),0.5)
     self.assertEqual(rbeis._df3(7,6,3),0.25)   
     self.assertEqual(rbeis._df3(7,2,3),1)

  # Check Exception is raised for df values other than 4, 5 or 6 
  def test_build_custom_df(self): 
      with self.assertRaises(Exception):
          rbeis._build_custom_df(7,{(1,1):2})

  def test_df4(self): 
     df4 = rbeis._build_custom_df(4,{(1,1):2,(2.2,3.3):8})
     self.assertEqual(df4(2.2,3.3,None),8.0)
     self.assertEqual(df4(2,3,None),1)
     self.assertEqual(df4(2,2,None),0)
     self.assertEqual(df4('abcd','abcd',None),0)
     self.assertEqual(df4(1.1,2.2,1),1)
     self.assertEqual(df4('abcd','bcde',None),1)
     
  def test_df5(self): 
     df5 = rbeis._build_custom_df(5,{(1,1):2,(2,3):8},3)
     self.assertEqual(df5(1,1,3),2)
     self.assertEqual(df5(1,4,3),0)    
     self.assertEqual(df5(1.1,5.5,3),1)
     self.assertEqual(df5(7,2,3),1)
     
  def test_df6(self): 
     df6 = rbeis._build_custom_df(6,{(1,1):2},3.0)
     self.assertEqual(df6(1,1,3),2.0)
     self.assertEqual(df6(5,5,3),0)
     self.assertEqual(df6(1,3,3),0.5)
     self.assertEqual(df6(7,6,3),0.25)   
     self.assertEqual(df6(7,2,3),1)

# Procedures to run after unit tests, if necessary
def tearDownModule():
    pass

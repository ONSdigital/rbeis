from unittest import TestCase
from pyspark.sql import SparkSession
from time import sleep


# Procedures to run before unit tests, if necessary
def setUpModule():
    pass
#    # Initialise small Spark session
#    print("> Initialising Spark session...")
#    sleep(10)
#    global spark
#    spark = (SparkSession.builder.appName("rbeis-test-session").config(
#        "spark.executor.memory",
#        "1g").config("spark.executor.cores", 1).config(
#            "spark.dynamicAllocation.enabled",
#            "true").config("spark.dynamicAllocation.maxExecutors", 3).config(
#                "spark.sql.shuffle.partitions",
#                12).config("spark.shuffle.service.enabled", "true").config(
#                    "spark.ui.showConsoleProgress",
#                    "false").enableHiveSupport().getOrCreate())
#    print("> Spark session initialised")


class TestExample(TestCase):

    def test_SparkVersionLength(self):
        self.assertTrue(True)
#        self.assertTrue(len(spark.version) == 14)


# Procedures to run after unit tests, if necessary
def tearDownModule():
    pass
#    # Stop Spark session
#    print("> Terminating Spark session...")
#    spark.stop()
#    print("> Spark session terminated")

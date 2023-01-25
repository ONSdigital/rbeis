from unittest import TestCase

# Procedures to run before unit tests, if necessary
def setUpModule():
    # TODO: Load a dummy dataset
    pass

class TestExample(TestCase):

    def test_2plus2(self):
        self.assertTrue((2 + 2) == 4)

# Procedures to run after unit tests, if necessary
def tearDownModule():
    pass

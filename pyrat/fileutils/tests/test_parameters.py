import unittest
from pyrat.fileutils.parameters import ParameterFile


class TestParameters(unittest.TestCase):

    def setUp(self):
        self.dummy_val = 5
        self.params = ParameterFile('./test_params.par')
        self.params_copy = self.params.copy()

    def testCopy(self):
        self.params_copy.range_samples = self.dummy_val
        self.assertNotEquals(self.params_copy.range_samples, self.params.range_samples)

    def testDotAccess(self):
        self.params.range_samples

    def testAttribute(self):
        self.params.file_title

    def testItemAccess(self):
        self.params['range_samples']


    def testAccessEquality(self):
        self.assertEqual(self.params['range_samples'], self.params.range_samples)


    def testDotSet(self):
        self.params.range_samples = self.dummy_val

    def testItemSet(self):
        self.params['range_samples'] = self.dummy_val

    def testAddItem(self):
        self.params['olo'] = 34

    def testSetEquality(self):
        self.params.range_samples = self.dummy_val
        self.params['range_samples'] = self.dummy_val
        self.assertEqual(self.params.range_samples, self.params['range_samples'])


    def testSaving(self):
        self.params.to_file('./test_save.par')


class TestRawParameters(unittest.TestCase):
    def setUp(self):
        self.dummy_val = 5
        self.params = ParameterFile('./raw_parameters.raw_par')


    # def testLoad(self):
    #     self.assertIn('CHP_num_samp', self.params)

    def testSaving(self):
        self.params.to_file('./test_save_raw.par')



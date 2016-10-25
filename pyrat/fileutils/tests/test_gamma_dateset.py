import unittest
from pyrat.fileutils.gpri_files import rawData


class TestRaw(unittest.TestCase):

    def setUp(self):
        self.raw_path = './20160614_090528.raw'
        self.raw = rawData(self.raw_path + '_par', self.raw_path)

    def testAddAttribute(self):
        self.raw.olo = 5

    def test_to_file(self):
        self.raw.tofile(self.raw_path + '_temp_par', self.raw_path + '_temp')
        new_raw = rawData(self.raw_path + '_temp_par', self.raw_path + '_temp')

    # def testDotAccess(self):
    #     self.params.range_samples
    #
    # def testAttribute(self):
    #     self.params.file_title
    #
    # def testItemAccess(self):
    #     self.params['range_samples']
    #
    #
    # def testAccessEquality(self):
    #     self.assertEqual(self.params['range_samples'], self.params.range_samples)
    #
    #
    # def testDotSet(self):
    #     self.params.range_samples = self.dummy_val
    #
    # def testItemSet(self):
    #     self.params['range_samples'] = self.dummy_val
    #
    # def testAddItem(self):
    #     self.params['olo'] = 34
    #
    # def testSetEquality(self):
    #     self.params.range_samples = self.dummy_val
    #     self.params['range_samples'] = self.dummy_val
    #     self.assertEqual(self.params.range_samples, self.params['range_samples'])
    #
    #
    # def testSaving(self):
    #     self.params.to_file('./test_save.par')
    #
    # def testKeyError(self):
    #     with self.assertRaises(KeyError):
    #         self.params['paolo']
    #
    # def testAttributeError(self):
    #     with self.assertRaises(AttributeError):
    #         self.params.paolo
    #
    # def testRepr(self):
    #     print(str(self.params))



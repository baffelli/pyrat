import unittest
from pyrat.fileutils.parameters import ParameterFile


class TestParameters(unittest.TestCase):

    def setUp(self):
        self.dummy_val = 5
        self.params = ParameterFile('../default_slc_par.par')
        self.params_copy = self.params.copy()
        self.copy_path = './copy.par'

    def testCopy(self):
        self.params_copy.range_samples = self.dummy_val
        self.assertNotEquals(self.params_copy.range_samples, self.params.range_samples)

    def testDotAccess(self):
        self.params.range_samples

    def testAttribute(self):
        self.params.file_title

    def testItemAccess(self):
        self.params['range_samples']

    def testItems(self):
        for item, value in self.params.items():
            print(value)

    def testContains(self):
        self.assertTrue('range_samples' in self.params)

    def testPop(self):
        self.params.pop('range_samples')
        self.assertFalse('range_samples' in self.params)

    def testAccessEquality(self):
        self.assertEqual(self.params['range_samples'], self.params.range_samples)


    def testDotSet(self):
        self.params.range_samples = self.dummy_val

    def testItemSet(self):
        self.params['range_samples'] = self.dummy_val

    def testAddAttribute(self):
        self.params.add_parameter('olo', 34)

    def testSetEquality(self):
        self.params.range_samples = self.dummy_val
        self.params['range_samples'] = self.dummy_val
        self.assertEqual(self.params.range_samples, self.params['range_samples'])

    def testFmtParsing(self):
        self.assertEqual(self.params.image_format, 'FCOMPLEX')

    def testSaving(self):
        self.params.file_title = "tezt + \n"
        self.params.tofile(self.copy_path)

    def testReload(self):
        self.params.tofile(self.copy_path)
        copy_params = ParameterFile(self.copy_path)
        self.assertEqual(self.params.image_format, copy_params.image_format)

    def testItems(self):
        print(self.params.items_with_unit())
        self.assertEqual(1,1)

    def testCreateEmptyAndAdd(self):
        params = ParameterFile({})
        params.add_parameter('image',23,'m')
        self.assertIn('image', params)

    def testKeyError(self):
        with self.assertRaises(KeyError):
            self.params['paolo']

    def testAttributeError(self):
        with self.assertRaises(AttributeError):
            self.params.paolo

    def testRepr(self):
        print(str(self.params))

class TestRawParameters(unittest.TestCase):
    def setUp(self):
        self.dummy_val = 5
        self.params = ParameterFile('./raw_parameters.raw_par')


    # def testLoad(self):
    #     self.assertIn('CHP_num_samp', self.params)

    def testSaving(self):
        self.params.tofile('./test_save_raw.par')



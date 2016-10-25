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



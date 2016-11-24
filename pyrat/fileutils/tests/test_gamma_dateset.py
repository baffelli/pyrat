import unittest

import numpy as np
from pyrat.fileutils.gpri_files import rawData, gammaDataset

import matplotlib.pyplot as plt

# class TestRaw(unittest.TestCase):

# def setUp(self):
#     self.raw_path = './20160614_090528.raw'
#     self.raw = rawData(self.raw_path + '_par', self.raw_path)
#
# def testAddAttribute(self):
#     self.raw.olo = 5
#
# def test_to_file(self):
#     self.raw.tofile(self.raw_path + '_temp_par', self.raw_path + '_temp')
#     new_raw = rawData(self.raw_path + '_temp_par', self.raw_path + '_temp')
#
#

class TestSLC(unittest.TestCase):
    def setUp(self):
        self.slc_path = './20150803_060749_AAAl.slc'
        self.slc = gammaDataset(self.slc_path + '.par', self.slc_path)

    def testSingleSlicing(self):
        slc_sl = self.slc[1]
        self.assertAlmostEqual(slc_sl.GPRI_az_start_angle, self.slc.GPRI_az_start_angle)
        self.assertAlmostEqual(slc_sl.GPRI_az_angle_step, self.slc.GPRI_az_angle_step)
        self.assertAlmostEqual(slc_sl.azimuth_lines, self.slc.azimuth_lines)

    def testRangeSlicing(self):
        start_r = 5
        slc_sl = self.slc[start_r:, :] * 1
        self.assertAlmostEqual(slc_sl.GPRI_az_start_angle, self.slc.GPRI_az_start_angle)
        self.assertAlmostEqual(slc_sl.near_range_slc, self.slc.near_range_slc + start_r * self.slc.range_pixel_spacing)

    def testAzimuthReversal(self):
        slc_sl = self.slc[:, ::-1]
        self.assertAlmostEqual(slc_sl.GPRI_az_angle_step, -self.slc.GPRI_az_angle_step)
        self.assertAlmostEqual(slc_sl.near_range_slc, self.slc.near_range_slc)
        self.assertAlmostEqual(slc_sl.near_range_slc, self.slc.near_range_slc)

    def testMask(self):
        slc_mask = np.ma.masked_array(self.slc)
        slc_mask.min()

    def testRaw(self):
        raw = rawData('/home/baffelli/PhD/trunk/Code/pyrat/pyrat/fileutils/tests/20160614_090528.raw_par',
                      '/home/baffelli/PhD/trunk/Code/pyrat/pyrat/fileutils/tests/20160614_090528.raw')
        #
        # def testDecimation(self):
        #     slc_sl = self.slc.decimate(5)
        #     self.assertAlmostEqual(slc_sl.GPRI_az_angle_step, self.slc.GPRI_az_angle_step *5)
        #     self.assertAlmostEqual(slc_sl.near_range_slc, self.slc.near_range_slc )
        #     self.assertAlmostEqual(slc_sl.near_range_slc, self.slc.near_range_slc)

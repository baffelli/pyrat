import unittest
import numpy as np
from pyrat.fileutils.gpri_files import rawData, gammaDataset, default_slc_dict, type_mapping
import matplotlib.pyplot as plt


class TestSLC(unittest.TestCase):

    def setUp(self):
        self.slc_par = default_slc_dict()
        #Generate fake slc data
        a = np.zeros([self.slc_par.range_samples,self.slc_par.azimuth_lines], dtype=type_mapping['FCOMPLEX'])
        # a += np.random.randn(*a.shape) + 1j * np.random.randn(*a.shape)
        self.slc = gammaDataset(self.slc_par, a)

    def testNan(self):
        a_1 = self.slc.copy()
        a_1[0:10,0:10] = np.nan
        a_1 / np.nanmax(a_1)

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

    def testRespectAttribute(self):
        self.slc.pane = '3'
        print(self.slc.__dict__)
        slc_slc = self.slc[:,::-1]
        self.assertEqual(self.slc.pane, slc_slc.pane)

    def testMask(self):
        slc_mask = np.ma.masked_array(self.slc)
        plt.imshow(np.abs(slc_mask))
        # plt.show()
        slc_mask.min()

    def testRaw(self):
        raw = rawData('/home/baffelli/PhD/trunk/Code/pyrat/pyrat/fileutils/tests/20160614_090528.raw_par',
                      '/home/baffelli/PhD/trunk/Code/pyrat/pyrat/fileutils/tests/20160614_090528.raw')

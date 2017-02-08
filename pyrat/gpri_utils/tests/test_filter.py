from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import skimage.data as data

import pyrat.fileutils.gpri_files as gpf
from .. import calibration as cal


class TestFilter(TestCase):
    def setUp(self):
        self.r_vec = np.linspace(0, 1000, num=100)
        self.az_vec = np.linspace(-10, 10, num=100)
        self.r_arm = 0.25
        self.r_ph = 0.012
        self.lam = gpf.C / 17e9

    def TestFilters(self):
        # az_vec_cal = np.linspace(-10,10,num=20)
        # rr1, aa1 = np.meshgrid(self.r_vec, az_vec_cal, indexing='ij')
        # rr, aa = np.meshgrid(self.r_vec, self.az_vec, indexing='ij')
        # filt, phase = cal.distance_from_phase_center(self.r_arm, self.r_ph, rr1, aa1, self.lam, wrap=False)
        image = data.camera().astype(np.complex64) * np.exp(1j * np.random.randn(*data.camera().shape))
        filter = np.zeros(13)
        filter[filter.shape[0] / 2 - 5:filter.shape[0] / 2 + 5] = np.exp(1j * np.linspace(-np.pi, np.pi, 10))
        filter = filter / np.sum(filter)
        filter = np.vstack((filter,) * image.shape[0])
        print(filter.shape)
        filtered = cal.filter1d(image.astype(np.complex64), filter)
        filtered1 = cal.filter2d(image.astype(np.complex64), filter)
        plt.subplot(3, 1, 1)
        plt.imshow(np.abs(filtered))
        ax = plt.gca()
        plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
        plt.imshow(np.abs(filtered1))
        plt.subplot(3, 1, 3, sharex=ax, sharey=ax)
        plt.imshow(np.angle(filtered * filtered1.conj()))
        plt.show()
        self.assertTrue(np.allclose(filtered, filtered1))

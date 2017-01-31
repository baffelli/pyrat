from unittest import TestCase

from skimage import data as dt
import numpy as np

from .. import  block_processing as bp

from .. import corefun as cf

import matplotlib.pyplot as plt

class TestBlock_array(TestCase):

    def testMean(self):
        data = dt.camera()
        print(data.shape)
        data_block = bp.block_array(data, [10,10], overlap=[6,3])
        data_block.process(lambda a: cf.smooth(a,[5,5]))
        plt.imshow(data_block.A)
        plt.show()

    def testSetAndGet(self):
        data = dt.checkerboard()
        data_block = bp.block_array(data, [25,25], overlap=[4,4])
        b = data_block[0]
        print(b.shape)
        data_block[6] = b * 4
        print(data_block.nblocks)
        f, (ax, ax_part) = plt.subplots(2,1)
        ax.imshow(data_block.A, vmin=0, vmax=255)
        ax_part.imshow(b, vmin=0, vmax=255)
        plt.show()
        # self.assertEqual(data_block[8],data_block[9])
        # data_mod = data_block.process(lambda a: 5* a)
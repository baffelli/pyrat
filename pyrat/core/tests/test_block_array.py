from unittest import TestCase

from skimage import data as dt
import numpy as np

from .. import  block_processing as bp

import matplotlib.pyplot as plt

class TestBlock_array(TestCase):

    def testMean(self):
        data = dt.checkerboard()
        data_block = bp.block_array(data, [25,25], overlap=[4,4])
        data_block.process(lambda a: a*5)

    def testSetAndGet(self):
        data = dt.checkerboard()
        data_block = bp.block_array(data, [25,25], overlap=[4,4])
        b = data_block[1]
        data_block[1] = b
        print(data_block.nblocks)
        f, (ax, ax_part) = plt.subplots(2,1)
        ax.imshow(data_block.A, vmin=0, vmax=255)
        ax_part.imshow(b, vmin=0, vmax=255)
        plt.show()
        self.assertEqual(data_block[8],data_block[9])
        # data_mod = data_block.process(lambda a: 5* a)
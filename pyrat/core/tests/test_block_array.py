from unittest import TestCase

from skimage import data as dt
import numpy as np

from .. import  block_processing as bp

from .. import corefun as cf

import matplotlib.pyplot as plt
import visualization.visfun as vf
import scipy.signal as sig

import scipy.ndimage as ndim
class TestBlock_array(TestCase):



    def testMean(self):
        def H(a):
            return cf.smooth(a,[3,3,3]).astype(a.dtype)
        data = dt.coffee()

        plt.show()
        data_block = bp.block_array(data, [11,11], overlap=[0,0], trim_output=True, pad_partial_blocks=False)
        A_proc = data_block.process(lambda a: H(a))
        print(A_proc.shape)
        print(data.shape)
        f, (a1,a2) = plt.subplots(2,1)
        a1.imshow(data)
        a2.imshow(A_proc)
        plt.show()

    # def testPadPartial(self):
    #     data = np.zeros((100, 100))
    #     data_block = bp.block_array(data.astype(np.double), [10, 10], overlap=[3, 3], pad_partial_blocks=True)
    #     shapes = []
    #     for b in data_block:
    #         shapes.append(b.data.shape)
    #     print(shapes)
    #
    # def testSetLowerEdge(self):
    #     data = np.zeros((100,100))
    #     val= 4
    #     data_block = bp.block_array(data.astype(np.double), [10, 13], overlap=[3,2])
    #     A_proc = data_block.process(lambda a: a.data + 4)
    #     plt.imshow(A_proc.A)
    #     plt.show()
    #     self.assertTrue(np.all(A_proc.A == val))

    def testGetSize(self):
        """
        Test if the block read from the
        data have the expected size
        Returns
        -------

        """
        data = dt.checkerboard()
        bs = [25,25]
        overlap = [4,4]
        data_block = bp.block_array(data, bs, overlap=overlap)
        b = data_block[5]
        self.assertEqual(list(b.shape), [b +o*2 for b,o in zip(bs,overlap)])

    def testSet(self):
        data = dt.checkerboard()
        bs = [25,25]
        overlap = [4,4]
        data_block = bp.block_array(data, bs, overlap=overlap)
        data_block[4] = data_block[4]

    def testSetAndGet(self):
        data = dt.checkerboard()
        data_block = bp.block_array(data, [25,25], overlap=[4,4])
        b = data_block[0]
        print(b.shape)
        # data_block[6] = b * 4
        print(data_block.nblocks)
        f, (ax, ax_part) = plt.subplots(2,1)
        ax.imshow(data_block.A, vmin=0, vmax=255)
        ax_part.imshow(b, vmin=0, vmax=255)
        plt.show()
        # self.assertEqual(data_block[8],data_block[9])
        # data_mod = data_block.process(lambda a: 5* a)
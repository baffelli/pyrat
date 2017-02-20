from unittest import TestCase

from skimage import data as dt
import numpy as np

from .. import  block_processing as bp

from .. import corefun as cf

import matplotlib.pyplot as plt
import visualization.visfun as vf
import scipy.signal as sig

import scipy.ndimage as ndim


import matplotlib.patches as patch
class TestBlock_array(TestCase):


    def testPartialPad(self):
        data = dt.coffee()
        win = [50,50]
        overlap = [10,10]
        total_size = [w + o*2 for w,o in zip(win,overlap)]
        data_block = bp.block_array(data, win, overlap=overlap,)
        sz = []
        f, (ax_full, ax_part, ax_proc) = plt.subplots(3,1)
        ax_full.imshow(data)
        ax_full.set_ylim(data.shape[0] + overlap[0],-overlap[0])
        ax_full.set_xlim(-overlap[1], data.shape[1] + overlap[1])
        r = patch.Rectangle([0,0],*total_size)
        ax_full.add_artist(r)
        for a in data_block:
            lower = a.location[0]*a.block_shape[0] - a.overlap[0]
            left = a.location[1]*a.block_shape[1] - a.overlap[1]
            r.set_xy([left, lower])
            ax_part.imshow(a.data)
            ax_proc.imshow(a.process(lambda a: a.data[:,:,::-1]))
            sz.append(a.data.shape)
            plt.draw()
            plt.pause(1)

    def testPlusOne(self):
        data = np.zeros((500,500))
        data = dt.checkerboard().astype(np.double)
        win = [11,11]
        overlap = [15,15]
        fun = lambda a: a.data +1
        data_block = bp.block_array(data, win, overlap=overlap, trim_output=True, pad_partial_blocks=False)
        data_proc = data_block.process(fun)
        f, (a1,a2) = plt.subplots(2,1, sharex=True, sharey=True)
        a1.imshow(data, vmin=0, vmax=256)
        a2.imshow(data_proc, vmin=0, vmax=256)
        plt.show()
        self.assertEqual(data.shape, data_proc.shape)

    def testMean(self):
        def H(a):
            return np.random.randint(255,*a.data.shape, dtype=np.int)
        data = dt.coffee()

        plt.show()
        data_block = bp.block_array(data, [6,6], overlap=[2,2], trim_output=True, pad_partial_blocks=False)
        A_proc = data_block.process(lambda a: H(a))
        print(A_proc.shape)
        print(data.shape)
        f, (a1,a2) = plt.subplots(2,1,sharex=True, sharey=True)
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
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 11:25:00 2014

@author: baffelli
"""
import numpy as np
import matplotlib
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
import pyrat
import pyrat.core
import pyrat.core.corefun
"""
Pyrat module for interferometric processing
"""

def get_shift(image1,image2, oversampling = 1, axes = (0,1)):
    pad_size = zip(np.zeros(image1.ndim),np.zeros(image1.ndim))
    for ax in axes:
        pad_size[ax] = (image1.shape[ax] * oversampling/2,image1.shape[ax] * oversampling/2)
    pad_size = tuple(pad_size)
    image1_pad = np.pad(image1,pad_size,mode='constant')
    image2_pad = np.pad(image2,pad_size,mode='constant')
    corr_image = norm_xcorr(image1_pad,image2_pad, axes = axes)
    shift = np.argmax(np.abs(corr_image))
    shift_idx = np.unravel_index(shift,corr_image.shape)
#    shift_idx = np.array(shift_idx) - corr_image.shape / 2
    shift_idx = tuple(np.divide(np.subtract(shift_idx , np.divide(corr_image.shape,2.0)),oversampling))
    return shift_idx, corr_image

def norm_xcorr(image1,image2, axes = (0,1)):
    image_1_hat = scipy.fftpack.fftn(image1, axes = axes)
    image_2_hat = scipy.fftpack.fftn(image2, axes = axes)
    phase_corr = scipy.fftpack.fftshift(scipy.fftpack.ifftn(image_1_hat * image_2_hat.conj() / (np.abs( image_1_hat * image_2_hat.conj())),axes = axes),axes= axes)
    return phase_corr

def patch_correlation(image1, image2, oversampling = 4, block_size = (10,10)):
    import itertools
    if image1.shape != image2.shape:
         raise ValueError("The two images do not have the same shape")
    else:
        #Split images
        B1 = pyrat.matrices.block_array(image1,block_size,[0,0])
        B2 = pyrat.matrices.block_array(image2,block_size,[0,0])
        shifts = np.zeros(B1.nblocks, dtype = np.complex64)
        for bl1,bl2 in itertools.izip(B1,B2):
            sh, corr = get_shift(bl1, bl2, oversampling = oversampling)
            idx = B1.center_index(B1.current)
            shifts[idx[0],idx[1]] = sh[0] + 1j*sh[1]
        return shifts

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 11:25:00 2014

@author: baffelli
"""
import numpy as _np
import scipy
import scipy.ndimage as ndim

from ..core import corefun as _cf

"""
Pyrat module for interferometric processing
"""



def estimate_coherence(ifgram, mli1, mli2, win, discard=True):
    """
    Estimates the coherence from a complex interferogram
    and two intensity images
    Parameters
    ----------
    ifgram
    mli1
    mli2

    Returns
    -------

    """
    cc = _cf.smooth(ifgram, win, discard=discard) / _np.sqrt((_cf.smooth(mli1, win, discard=discard) * _cf.smooth(mli2, win, discard=discard)))
    # cc_outliers = _np.abs(cc) > 1
    # cc[cc_outliers] =  1 * _np.exp(1j* _np.angle(cc[cc_outliers]))
    return cc

#
# def get_shift(image1, image2, oversampling=1, axes=(0, 1)):
#     pad_size = zip(np.zeros(image1.ndim), np.zeros(image1.ndim))
#     for ax in axes:
#         pad_size[ax] = (image1.shape[ax] * oversampling / 2, image1.shape[ax] * oversampling / 2)
#     pad_size = tuple(pad_size)
#     image1_pad = np.pad(image1, pad_size, mode='constant')
#     image2_pad = np.pad(image2, pad_size, mode='constant')
#     corr_image = norm_xcorr(image1_pad, image2_pad, axes=axes)
#     shift = np.argmax(np.abs(corr_image))
#     shift_idx = np.unravel_index(shift, corr_image.shape)
#     #    shift_idx = np.array(shift_idx) - corr_image.shape / 2
#     shift_idx = tuple(np.divide(np.subtract(shift_idx, np.divide(corr_image.shape, 2.0)), oversampling))
#     return shift_idx, corr_image
#
#
# def norm_xcorr(image1, image2, axes=(0, 1)):
#     image_1_hat = scipy.fftpack.fftn(image1, axes=axes)
#     image_2_hat = scipy.fftpack.fftn(image2, axes=axes)
#     phase_corr = scipy.fftpack.fftshift(
#         scipy.fftpack.ifftn(image_1_hat * image_2_hat.conj() / (np.abs(image_1_hat * image_2_hat.conj())), axes=axes),
#         axes=axes)
#     return phase_corr


def patch_correlation(image1, image2, oversampling=4, block_size=(10, 10)):
    import itertools
    if image1.shape != image2.shape:
        raise ValueError("The two images do not have the same shape")
    else:
        # Split images
        B1 = pyrat.matrices.block_array(image1, block_size, [0, 0])
        B2 = pyrat.matrices.block_array(image2, block_size, [0, 0])
        shifts = np.zeros(B1.nblocks, dtype=np.complex64)
        for bl1, bl2 in itertools.izip(B1, B2):
            sh, corr = get_shift(bl1, bl2, oversampling=oversampling)
            idx = B1.center_index(B1.current)
            shifts[idx[0], idx[1]] = sh[0] + 1j * sh[1]
        return shifts


def hp_filter(ifgram, ws):
    kernel = kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]]) / 9
    ifgram_filt = ndim.convolve(ifgram.real, kernel) + 1j * ndim.convolve(ifgram.imag, kernel)
    return ifgram_filt


def compute_baseline(slc1, slc2):
    bl = slc1.phase_center - slc2.phase_center


def itab(n_slc, window, stride, step, n_ref):
    tab = []
    #list with reference numbers
    if stride == 0:
        reference = [n_ref]
        window = 0
    else:
        reference = list(range(n_ref,n_slc, stride))
    counter = 1
    for master in reference:
        for slave in range(master + step, master+ step+window, step):
            if slave < n_slc:
                line = [master, slave, counter]
                counter += 1
                tab.append(line)
    return tab

def itab_to_incidence_matrix(itab):
    n_slc = _np.max(_np.array(itab)[:,0:2])
    A = _np.zeros((len(itab),n_slc+1))
    for idx_master, idx_slave, idx_itab, *rest in itab:
        print(idx_itab)
        A[idx_itab - 1, idx_master] = 1
        A[idx_itab - 1, idx_slave] = -1
    return A
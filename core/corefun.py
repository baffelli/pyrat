# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:34:25 2014

@author: baffelli
"""
import numpy as np
from scipy import ndimage
def outer_product(data):
    """
    Computes the outer product of multimensional data along the last dimension
    
    Parameters
    ----------
    data : ndarray
        The data to process
    Returns
    -------
    ndarray  
        The outer product array
    """
    if data.ndim > 1:
        T = np.einsum("...i,...j->...ij",data,data.conjugate())
    else:
        T = np.outer(data,data)
    return T
    
def smooth(T, window):
    """
    Smoothes data using a multidmensional boxcar filter 
    Parameters
    ----------
    T : ndarray
        The data to process
    window : iterable
        The size of the smoothing window
    Returns
    -------
    ndarray
        the smoothed array
    """
    T_out_real = ndimage.filters.uniform_filter(T.real,window)
    T_out_imag = ndimage.filters.uniform_filter(T.imag,window)
    T_out = T_out_real  + 1j * T_out_imag
    return T_out
    
def shift_array(array,shift):
    """
    Shift a multidimensional array and pads it with zeros
    Parameters
    ----------
    array : ndarray
        the array to shift
    window  : tuple
        a tuple of shifts
    Returns
    -------
    ndarray : 
        the shifted array
    """
    pre_shifts = tuple()
    post_shifts = tuple()
    index_list = tuple()
    for current_shift in shift:
        if current_shift > 0:
            pre_shifts = pre_shifts + (abs(current_shift),)
            post_shifts = post_shifts + (0,)
        else:
            pre_shifts = pre_shifts + (0,)
            post_shifts = post_shifts + (abs(current_shift),)
        if current_shift is 0:
            index_list = index_list + (Ellipsis,)
        elif current_shift > 0:
            index_list = index_list + (slice(None,-current_shift),)
        elif current_shift < 0:
            index_list = index_list + (slice(abs(current_shift),None),)
    shifts = zip(pre_shifts,post_shifts)
    array_1 = np.pad(array,tuple(shifts),mode='constant',constant_values = (0,))[index_list]
    return array_1
    
def is_hermitian(T):
    """
    Checks if a multidimensional array is composed of subarray that are hermitian matrices along the last
    two dimensions
    Parameters
    ----------
    T : ndarray
        the array to test
    Returns
    -------
    bool :
        A flag indicating the result
    """
    try:
        shp = T.shape
        thresh = 1e-6
        if T.shape == (3,3) and np.max(T.transpose().conjugate() - T) < thresh:
            is_herm = True
        elif shp[2:4] == (3,3) and np.max(T.transpose([0,1,3,2]).conjugate() - T) < thresh:
            is_herm = True
        elif T.ndim is 3 and np.max(T.transpose([0,2,1]).conjugate() - T)< thresh:
            is_herm = True
        else:
            is_herm = False
        return is_herm
    except (TypeError, AttributeError):
        print("T is not a numpy array")
    
def transform(A,B,C):
    """
    This function transforms the matrix B by the matrices A and B
    
    Parameters
    ----------
    A : ndarray
        the premultiplication matrix
    B : ndarray
        the matrix to be transformed
    C : ndarray
        the postmultiplication matrix
    Returns
    -------
    out : ndarray
        the transformed matrix
    """
    out = np.einsum("...ik,...kl,...lj->...ij",A,B,C)
    return out

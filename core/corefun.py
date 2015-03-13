# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:34:25 2014

@author: baffelli
"""
import numpy as _np
import scipy as _sp
from scipy import ndimage as _nd
from numpy.lib.stride_tricks import as_strided as _ast

def outer_product(data,data1, large = False):
    """
    Computes the outer product of multimensional data along the last dimension.
    
    Parameters
    ----------
    data : ndarray
        The data to process.
    Returns
    -------
    ndarray  
        The outer product array.
    """
    if ~large:
        if data.ndim > 1:
            T = _np.einsum("...i,...j->...ij",data,data1.conjugate())
        else:
            T = _np.outer(data,data1.conjugate())
        return T

    


def smooth(T, window, fun = _nd.filters.uniform_filter ):
    """
    Smoothes data using a multidmensional boxcar filter .
    
    Parameters
    ----------
    T : ndarray
        The data to process.
    window : iterable
        The size of the smoothing window.
    fun : function
        The function used for smoothing
    Returns
    -------
    ndarray
        the smoothed array.
    """
    T1 = _np.array(T[:])
    T1[_np.isnan(T1)] = 0
    T1_out_real = fun(T1.real,window)
    T1_out_imag = fun(T1.imag,window)
    T1_out = T1_out_real  + 1j * T1_out_imag
    T1_out[_np.isnan(T)] = _np.nan
    return T.__array_wrap__(T1_out)
    


def smooth_1(T,window):
    T_sm = T * 1
    for ax, roll_size in enumerate(window):
        T_sm = T_sm + _np.roll(T, roll_size, axis = ax)
    return T_sm / _np.prod(window)
    
def shift_array(array,shift):
    """
    Shift a multidimensional array and pads it with zeros.
    
    Parameters
    ----------
    array : ndarray
        the array to shift.
    window  : tuple
        a tuple of shifts.
    Returns
    -------
    ndarray : 
        the shifted array
    """
    array_1 = array * 1
    for ax, current_shift in enumerate(shift):
        array_1 = _np.roll(array_1, current_shift, axis = ax)
    return array.__array_wrap__(array_1)
    
def is_hermitian(T):
    """
    Checks if a multidimensional array is composed of subarray that are hermitian matrices along the last
    two dimensions.
    
    Parameters
    ----------
    T : ndarray
        the array to test.
    Returns
    -------
    bool :
        A flag indicating the result.
    """
    try:
        shp = T.shape
        thresh = 1e-6
        if T.shape == (3,3) and _np.nanmax(T.transpose().conjugate() - T) < thresh:
            is_herm = True
        elif shp[2:4] == (3,3) and _np.nanmax(T.transpose([0,1,3,2]).conjugate() - T) < thresh:
            is_herm = True
        elif T.ndim is 3 and _np.nanmax(T.transpose([0,2,1]).conjugate() - T)< thresh:
            is_herm = True
        else:
            is_herm = False
        return is_herm
    except (TypeError, AttributeError):
        print("T is not a numpy array")
    
def transform(A,B,C):
    """
    This function transforms the matrix B by the matrices A and B.
    
    Parameters
    ----------
    A : ndarray
        the premultiplication matrix.
    B : ndarray
        the matrix to be transformed.
    C : ndarray
        the postmultiplication matrix.
    Returns
    -------
    out : ndarray
        the transformed matrix
    """
    out = B.__array_wrap__(_np.einsum("...ik,...kl,...lj->...ij",A,B,C))
    return out


def matrix_root(A):
    l,w = (_np.linalg.eig(_np.array(A)))
    l_sq = (_np.sqrt(l))
    if A.ndim > 2:
        L_sq = _np.zeros_like(w)
        for idx_diag in range(w.shape[2]):
            L_sq[:,:,idx_diag,idx_diag] = l_sq[:,:,idx_diag]
    else:
        L_sq = _np.diag(l_sq)
    A_sq = transform(w, L_sq, _np.linalg.inv(w))
    return A_sq
    

def window_idx(arr, idx, zs):
    indices = []
    for cnt, (current_size,current_idx) in enumerate(zip(zs, idx)):
        mi = _np.clip(current_idx - current_size, 0, arr.shape[cnt])
        mx = _np.clip(current_idx + current_size, 0, arr.shape[cnt])
        sl = slice(mi, mx)
        indices.append(sl)
    return indices 
    


def block_view(A, block= (3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    #Compute padding for block
    pad = [block_dim - dim % block_dim for dim, block_dim in zip(A.shape,block)]
    pad = pad + (len(A.shape) - len(block)) * [0]
    total_pad = [(p,0) for p in pad]
    print total_pad
    A = _np.pad(A,total_pad,mode = 'constant')
    print A.shape
    shape= (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block + A.shape[len(block)::] 
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return _ast(A, shape= shape, strides= strides)
    
    
def block_to_array(A, block = (3,3)):
    new_shape = (A.shape[0] * block[0], A.shape[1] * block[1]) + A.shape[len(block) + 2::]
    strides = A.strides[len(block)::]
    print (new_shape), (strides)
    return _ast(A, shape= new_shape, strides= strides)
    

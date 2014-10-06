# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:34:25 2014

@author: baffelli
"""
import numpy as np
import scipy
from scipy import ndimage, fftpack
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
            T = np.einsum("...i,...j->...ij",data,data1.conjugate())
        else:
            T = np.outer(data,data1.conjugate())
        return T
    else:
        mulop = np.multiply
        it = np.nditer([x, y, out], ['external_loop'],
                [['readonly'], ['readonly'], ['writeonly', 'allocate']],
                op_axes=[range(x.ndim)+[-1]*y.ndim,
                         [-1]*x.ndim+range(y.ndim),
                         None])
    
        for (a, b, c) in it:
            mulop(a, b, out=c)

    return it.operands[2]
    


def smooth(T, window, fun = ndimage.filters.uniform_filter ):
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
    T1 = T[:]
    T1[np.isnan(T1)] = 0
    T1_out_real = fun(T1.real,window)
    T1_out_imag = fun(T1.imag,window)
    T1_out = T1_out_real  + 1j * T1_out_imag
    T1_out[np.isnan(T)] = np.nan
    return T1_out

def smooth_1(T,window):
    T_sm = T * 1
    for ax, roll_size in enumerate(window):
        T_sm = T_sm + np.roll(T, roll_size, axis = ax)
    return T_sm / np.prod(window)
    
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
    array_1 = np.pad(array,tuple(shifts),mode='constant',constant_values = (1,))[index_list]
    return array_1
    
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
    out = B.__array_wrap__(np.einsum("...ik,...kl,...lj->...ij",A,B,C))
    return out


def matrix_root(A):
    l,w = (np.linalg.eig(np.array(A)))
    l_sq = (np.sqrt(l))
    if A.ndim > 2:
        L_sq = np.zeros_like(w)
        for idx_diag in range(w.shape[2]):
            L_sq[:,:,idx_diag,idx_diag] = l_sq[:,:,idx_diag]
    else:
        L_sq = np.diag(l_sq)
    A_sq = transform(w, L_sq, np.linalg.inv(w))
    return A_sq
    
def range_variant_filter(data,area):
    """
    This function implements a moving average 
    filter with variable size, in order to obtain
    a filtered pixel with uniform size.
    
    Parameters
    ----------
    data : ndarray
        the data to process.
    area : double
        the desired pixel area in meters.
    """
    filtered = data * 1
    filter_size = np.zeros(data.shape[0:2])
    az_step = data.az_vec[1] - data.az_vec[0]
    r_step = data.r_vec[1] - data.r_vec[0]
    for idx_r in np.arange(data.shape[1]):
        n_r = idx_r + 1
        pixel_area = az_step * ((n_r*r_step)**2 - ((n_r -1) * r_step)**2)
        n_pix = np.ceil(area  / pixel_area)
        current_pixels = data[:,idx_r,:,:]
        filtered[:,idx_r,:,:] = smooth(current_pixels,[n_pix,1,1])
        filter_size[:,idx_r] = n_pix
    return filtered, filter_size

def split_bandwdith(data,n_splits,axis = 0):
    pad_size = n_splits - data.shape[axis] % n_splits
    print pad_size
    data_hat = scipy.fftpack.fftshift(scipy.fftpack.fft(data,axis = axis),axes = (axis,))
    pad_arr = [[0,0]] * data.ndim
    pad_arr[axis] = [pad_size,0]
    data_hat = np.pad(data_hat,pad_arr,mode=  'constant')
    print data_hat.shape
    data_split = np.split(data_hat,n_splits,axis = axis)
    data_cube = []
    broadcast_slice = [None,] * data.ndim
    broadcast_slice[axis] = Ellipsis
    for data_slice in data_split:
        data_win = np.hamming(data_slice.shape[axis])[broadcast_slice] * data_slice
        data_cube = data_cube + [ scipy.fftpack.ifft(scipy.fftpack.ifftshift(data_win, axes = (axis,)),axis = axis),]
    return data_cube
        
    
from numpy.lib.stride_tricks import as_strided as ast


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
    A = np.pad(A,total_pad,mode = 'constant')
    print A.shape
    shape= (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block + A.shape[len(block)::] 
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)
    
    
def block_to_array(A, block = (3,3)):
    new_shape = (A.shape[0] * block[0], A.shape[1] * block[1]) + A.shape[len(block) + 2::]
    strides = A.strides[len(block)::]
    print (new_shape), (strides)
    return ast(A, shape= new_shape, strides= strides)
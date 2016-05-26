# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:34:25 2014

@author: baffelli
"""
import numpy as _np
import scipy as _sp
from scipy import ndimage as _nd
from numpy.lib.stride_tricks import as_strided as _ast
import tempfile as _tf
import subprocess as _sp
import os as _os
from ..fileutils import gpri_files as _gpf
import scipy.ndimage as _ndim
#Set environment variables
_os.environ['GAMMA_HOME']='/usr/local/GAMMA_SOFTWARE-20130717'
_os.environ['ISP_HOME']=_os.environ['GAMMA_HOME'] + '/ISP'
_os.environ['MSP_HOME']=_os.environ['GAMMA_HOME'] + '/MSP'
_os.environ['DIFF_HOME']=_os.environ['GAMMA_HOME'] + '/DIFF'
_os.environ['GEO_HOME']=_os.environ['GAMMA_HOME'] + '/GEO'
_os.environ['LD_LIBRARY_PATH']=_os.environ['GAMMA_HOME'] +'/lib'
_os.environ["PATH"] = _os.environ["PATH"] +  _os.pathsep + _os.environ['GAMMA_HOME'] + '/bin' + _os.pathsep + _os.environ['ISP_HOME'] + '/bin' + _os.pathsep + _os.environ['DIFF_HOME'] + '/bin'



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
    T1_out[_np.isnan(T)] = 0
    return T.__array_wrap__(T1_out)
    


def smooth_1(T,window):
    T_sm = T * 1
    for ax, roll_size in enumerate(window):
        T_sm = T_sm + _np.roll(T, roll_size, axis = ax)
    return T_sm / _np.prod(window)


def multi_look(dataset, lks):
    slc_par_tf, slc_tf = dataset.to_tempfile()
    mli_par, mli = _gpf.temp_dataset()
    cmd_list = [
        "multi_look",
        slc_tf.name,
        slc_par_tf.name,
        mli.name,
        mli_par.name,
        lks[0],
        lks[1],
        '-',
        '-'
    ]
    try:
        cmd = " ".join([str(i) for i in cmd_list])
        status = _sp.Popen(cmd, env=_os.environ, stderr=_sp.STDOUT)
    except _sp.CalledProcessError as e:
        print(e.output)
    multi_looked = _gpf.gammaDataset(mli_par.name, mli.name)
    return multi_looked


def unwrap(intf, wgt, mask):
    #Write to a binary file
    intf_tf = _gpf.temp_binary()
    intf.T.astype(_np.dtype('>c8')).tofile(intf_tf.name,'')
    #Temporary file for the unwrapped
    unw_tf = _gpf.temp_binary()
    #Temporary file for the mask
    mask_tf = _gpf.temp_binary(suffix='.bmp')
    _gpf.to_bitmap(mask.T, mask_tf)
    #Temporary file for the wgt
    wgt_tf = _gpf.temp_binary()
    wgt.T.astype(_gpf.type_mapping['FLOAT']).tofile(wgt_tf.name)
    arg_list = [
        'mcf',
        intf_tf.name,
        wgt_tf.name,
        mask_tf.name,
        unw_tf.name,
        str(intf.shape[0]),
        1
    ]
    try:
        cmd = [str(el) for el in arg_list]
        P = _sp.Popen(cmd,env=_os.environ, stderr=_sp.STDOUT)
        P.wait()
    except Exception as e:
        print(e)
        return -1
    unwrapped = _np.fromfile(unw_tf.name, dtype=_gpf.type_mapping['FLOAT']).reshape(intf.shape[::-1]).T
    return unwrapped


def ptarg(slc, ridx, azidx,rwin=32, azwin=64, osf=16, ):
    complex_interp = lambda arr, osf : _ndim.interpolation.zoom(arr.real, osf) + 1j * _ndim.interpolation.zoom(arr.imag, osf)
    sw = 4
    #Add one ellispis in case of a ndimensional image
    additional_dim = slc.ndim - 2 if (slc.ndim - 2) >= 0 else 0
    search_win = (slice(ridx - sw / 2, ridx + sw / 2),
           slice(azidx - sw / 2, azidx + sw / 2), ) + (None,)*additional_dim
    print(search_win)
    print(slc.shape)
    # Find the maxium
    ptarg = slc[search_win]
    mx = _np.argmax(_np.abs(ptarg))
    mx_list = _np.unravel_index(mx, ptarg.shape)
    mx_r, mx_az = mx_list[0:2]

    #Maximum in global system
    mx_r_glob = mx_r + ridx
    mx_az_glob = mx_r + azidx
    # New window
    win_1 = (slice(mx_r_glob - rwin / 2, mx_r_glob + mx_r + rwin / 2),
             slice(mx_az_glob - azwin / 2, mx_az_glob + azwin / 2),)
    mx_sample = (slc[(mx_r_glob, mx_az_glob) + (Ellipsis,)*additional_dim])
    ptarg = slc[win_1]
    if slc.ndim == 2:
        ptarg_zoom = complex_interp(ptarg, osf)
    else:
        if slc.ndim == 3:
            ptarg_zoom = []
            for i in range(slc.shape[-1]):
                ptarg_zoom[i] = complex_interp(ptarg[:,:,i], osf)
        elif slc.ndim == 4:
            ptarg_zoom = [[0 for x in range(ptarg.shape[-2])] for y in range(ptarg.shape[-2])]
            for i in range(slc.shape[-1]):
                for j in range(slc.shape[-2]):
                    ptarg_zoom[i][j] = complex_interp(ptarg[:,:,i,j], osf)
            ptarg_zoom = _np.array(ptarg_zoom).transpose([2,3,0,1])
    mx_zoom = _np.argmax(_np.abs(ptarg_zoom))
    mx_list_zoom = _np.unravel_index(mx_zoom, ptarg_zoom.shape)
    mx_r_zoom, mx_az_zoom = mx_list_zoom[0:2]
    rplot = ptarg_zoom[(Ellipsis, mx_az_zoom) + (Ellipsis,)*additional_dim]
    azplot = ptarg_zoom[(mx_r_zoom, Ellipsis) + (Ellipsis,)*additional_dim]
    return ptarg_zoom, rplot, azplot, (mx_r_zoom, mx_az_zoom)


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
    pad_array = [(0,0),] * array.ndim
    for ax, current_shift in enumerate(shift):
        slice_array = [Ellipsis,] * array.ndim
        if shift > 0:
            pad_array[ax] = (0,shift)
            slice_array[ax] = slice(shift)
        else:
            pad_array[ax] = (_np.abs(shift),0)
            slice_array[ax] = slice(0,-shift)
    array_1_pad = _np.pad(array_1, tuple(pad_array), mode='const')
    array_1 = _np.roll(array_1_pad, current_shift, axis = ax)[tuple[slice_array]]
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
        thresh = 1e-4
        # if T.shape[0] == T.shape[1] and _np.nanmax(T.transpose().conjugate() - T) < thresh:
        #     is_herm = True
        # elif T.nidm is 4 and _np.nanmax(_np.abs(T - T.transpose([0,1,3,2]).conjugate())) < thresh:
        #     print('Here')
        #     is_herm = True
        # elif T.ndim is 3 and _np.nanmax(T.transpose([0,2,1]).conjugate() - T)< thresh:
        #     is_herm = True
        # else:
        #     is_herm = False
        dims = range(T.ndim)
        dims[-2], dims[-1] = dims[-1], dims[-2]
        T_H = T.transpose(dims).conj()
        if _np.nanmax(_np.abs(T_H- T).flatten()) < thresh:
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

def maximum_around(arr, idx, ws):
    """
    Finds the maximum of an array around a given index for a given
    search window size
    Parameters
    ----------
    arr : ndarray
        The array where to find the maximum
    idx : iterable of int or int
        The index around which to find the maximum
    ws  : iterable of int or int
        The window size where to search for the maximum

    Returns
    -------
        iterable of int or int

    """
    #Get indices to slice the array
    indices = window_idx(arr, idx, ws)
    #Slice the array
    arr_section = arr[indices]
    #Find the argmax
    max_idx = _np.argmax(arr_section)
    #Now that we have the indices, we convert it
    #to tuple
    max_idx = _np.unravel_index(max_idx, arr_section.shape)
    #We have now to convert it into the "coordinates" of the big array
    final_idx = [local_idx + global_idx - current_size for
                 local_idx, global_idx, current_size in zip(max_idx, idx, ws)]
    return final_idx


def split_equal(A, split_size, axis=0):
    """
    Splits a n-dimensional array into sub-arrays of equal size.
    Parameters
    ----------
    A   : ndarray
        The array to be split into blocks
    split_size : int
        The number of blocks
    axis : int, optional
        The axis along which to split
    Returns
        list of ndarrays
    -------
    padding = (a.shape[axis])%split_size
    return np.split(np.concatenate((A,np.zeros(padding))),split_size)

    """


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
    A = _np.pad(A,total_pad,mode = 'constant')
    shape= (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block + A.shape[len(block)::] 
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return _ast(A, shape= shape, strides= strides)
    
    
def block_to_array(A, block = (3,3)):
    new_shape = (A.shape[0] * block[0], A.shape[1] * block[1]) + A.shape[len(block) + 2::]
    strides = A.strides[len(block)::]
    return _ast(A, shape= new_shape, strides= strides)
    

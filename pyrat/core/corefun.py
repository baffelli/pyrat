# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:34:25 2014

@author: baffelli
"""
import os as _os
import subprocess as _sp

import numpy as _np
import scipy as _sp
import scipy.ndimage as _ndim
from numpy.lib.stride_tricks import as_strided as _ast
from scipy import ndimage as _nd
from scipy.interpolate import splrep, sproot

from ..fileutils import gpri_files as _gpf
from ..geo import geofun as _gf


def outer_product(data, data1, large=False):
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
            T = _np.einsum("...i,...j->...ij", data, data1.conjugate())
        else:
            T = _np.outer(data, data1.conjugate())
        return T


def dB(arr, power=True):
    factor = 10 if power else 20
    return factor * _np.log10(_np.abs(arr))


def fspf(T, window):
    T_gc, x_vec, y_vec = _gf.geocode_image(T, 1)
    T_sm = smooth(T_gc, window)


def smooth(T, window, fun=_nd.filters.uniform_filter):
    """
    Smoothes data using a multidmensional boxcar filter .
    
    Parameters
    ----------
    T : numpy.ndarray
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
    T1_out_real = fun(T1.real, window)
    T1_out_imag = fun(T1.imag, window)
    T1_out = T1_out_real + 1j * T1_out_imag
    T1_out[_np.isnan(T)] = 0
    return T.__array_wrap__(T1_out)


def smooth_1(T, window):
    T_sm = T * 1
    for ax, roll_size in enumerate(window):
        T_sm = T_sm + _np.roll(T, roll_size, axis=ax)
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
    # Write to a binary file
    intf_tf = _gpf.temp_binary()
    intf.T.astype(_np.dtype('>c8')).tofile(intf_tf.name, '')
    # Temporary file for the unwrapped
    unw_tf = _gpf.temp_binary()
    # Temporary file for the mask
    mask_tf = _gpf.temp_binary(suffix='.bmp')
    _gpf.to_bitmap(mask.T, mask_tf)
    # Temporary file for the wgt
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
        P = _sp.Popen(cmd, env=_os.environ, stderr=_sp.STDOUT)
        P.wait()
    except Exception as e:
        print(e)
        return -1
    unwrapped = _np.fromfile(unw_tf.name, dtype=_gpf.type_mapping['FLOAT']).reshape(intf.shape[::-1]).T
    return unwrapped


def FWHM(curve):
    x = _np.arange(-len(curve) / 2, len(curve) / 2)
    y = _np.abs(curve) - _np.max(_np.abs(curve)) / 2
    spline = splrep(x, y)
    r = sproot(spline)
    if len(r) > 2:
        raise ArithmeticError("More than two peaks, no FWHM found.")
    elif len(r) < 2:
        raise ArithmeticError("No peak found, FWHM cannot be determined.")
    else:
        return abs(r[1] - r[0])


# TODO implement
def half_power_indices(data):
    """
    Returns the indices in the data that correspond to the half power beamwidth
    Parameters
    ----------
    data

    Returns
    -------

    """
    return 0


def complex_interp(arr, osf, polar=False):
    if polar:
        return _ndim.interpolation.zoom(arr.real, osf) + 1j * _ndim.interpolation.zoom(arr.imag,
                                                                                       osf)
    else:
        return _ndim.interpolation.zoom(_np.abs(arr), osf) * _np.exp(1j * _ndim.interpolation.zoom(_np.angle(arr), osf))


def ptarg(slc, ridx, azidx, rwin=32, azwin=64, osf=16, sw=(2, 4)):
    """
    Point target analysis.
    Parameters
    ----------
    slc : scatteringMatrix, coherencyMatrix
        The dataset from which to extract the point target response
    ridx : int
        Range index of the point target
    azidx : int
        Azimuth index of the point target
    rwin : int, optional
        Range size of window around point target
    azwin : int, optional
        Azimuth  size of window around point target
    osf : int, optional
        Oversampling factor
    sw : int, optional
        Seach window to find the maximum around (`ridx`, `azidx`)

    Returns
    -------
    ndarray
        Oversampled point target response
    ndarray
        Oversampled range plot
    ndarray
        Oversampled azimuth plot
    iterable
        Location of maxiumum in oversampled response

    """
    # lambda function for complex interpolation
    # complex_interp = lambda arr, osf: _ndim.interpolation.zoom(arr.real, osf) + 1j * _ndim.interpolation.zoom(arr.imag,
    #                                                                                                           osf)
    # Add one ellispis in case of a ndimensional image
    additional_dim = slc.ndim - 2 if (slc.ndim - 2) >= 0 else 0
    # In the additional dimensions, do not use a search window ("we have a sort of "well" in the data cube)
    sw = sw + (1,) * additional_dim
    mx_glob = maximum_around(_np.abs(slc), [ridx, azidx], sw)
    # search_win = (slice(ridx - sw[0] / 2, ridx + sw[0] / 2),
    #               slice(azidx - sw[1] / 2, azidx + sw[1] / 2),) + (None,) * additional_dim
    # # Find the maxium whitin the search window
    # slc_section = slc[search_win]
    # mx = _np.argmax(_np.abs(slc_section))
    # mx_list = _np.unravel_index(mx, slc_section.shape)
    # mx_r, mx_az = mx_list[0:2]
    # # Maximum in global system
    # mx_r_glob = mx_r + ridx - sw[0] / 2
    # mx_az_glob = mx_az + azidx - sw[1] / 2
    # New window
    win_1 = window_idx(slc, mx_glob, [rwin, azwin])
    # limits = [(_np.clip(mx - win // 2, 0, shp), _np.clip(mx + win // 2, 0, shp)) for mx, win, shp in
    #           zip(mx_glob, [rwin, azwin], slc.shape)]  # limits for search window
    #
    # win_1 = (slice(limits[0][0], limits[0][1]),
    #          slice(limits[1][0], limits[1][1]),)
    slc_section = slc[win_1]  # slice aroudn global maximum
    if slc.ndim == 2:
        ptarg_zoom = complex_interp(slc_section, osf, polar=True)
    else:
        if slc.ndim == 3:
            ptarg_zoom = []
            for i in range(slc.shape[-1]):
                ptarg_zoom[i] = complex_interp(slc_section[:, :, i], osf)
        elif slc.ndim == 4:
            ptarg_zoom = [[0 for x in range(slc_section.shape[-2])] for y in range(slc_section.shape[-2])]
            for i in range(slc.shape[-1]):
                for j in range(slc.shape[-2]):
                    ptarg_zoom[i][j] = complex_interp(slc_section[:, :, i, j], osf)
            ptarg_zoom = _np.array(ptarg_zoom).transpose([2, 3, 0, 1])
    mx_zoom = _np.argmax(_np.abs(ptarg_zoom))
    mx_list_zoom = _np.unravel_index(mx_zoom, ptarg_zoom.shape)
    print(ptarg_zoom.shape)
    mx_r_zoom, mx_az_zoom = mx_list_zoom[0:2]
    rplot = ptarg_zoom[(Ellipsis, mx_az_zoom) + (Ellipsis,) * additional_dim]
    azplot = ptarg_zoom[(mx_r_zoom, Ellipsis) + (Ellipsis,) * additional_dim]
    try:
        # analyse resolution if slc has azimuth and range parameters
        if ptarg_zoom.ndim == 2:
            rplot_analysis = rplot
            azplot_analysis = azplot
        else:
            rplot_analysis = rplot[0, 0]
            azplot_analysis = azplot[0, 0]

        az_spacing = slc.GPRI_az_angle_step[0] / osf
        r_spacing = slc.range_pixel_spacing[0] / osf
        # mx_val = _np.abs(ptarg_zoom)[(mx_r_zoom, mx_az_zoom)]
        # range resolution
        # Half power length
        try:
            hpbw_r = FWHM(_np.abs(rplot_analysis) ** 2) * r_spacing
        except:
            hpbw_r = 0
        try:
            hpbw_az = FWHM(_np.abs(azplot_analysis) ** 2) * az_spacing
        except:
            hpbw_az = 0
        res_dict = {'range_resolution': [hpbw_r, 'm'], 'azimuth_resolution': [hpbw_az, 'deg']}
        # Construct range and azimuth vector
        r_vec = _np.arange(-ptarg_zoom.shape[0] / 2, ptarg_zoom.shape[0] / 2) * r_spacing
        az_vec = _np.arange(-ptarg_zoom.shape[1] / 2, ptarg_zoom.shape[1] / 2) * az_spacing
    except AttributeError:
        res_dict = {}
        r_vec = []
        az_vec = []

    return ptarg_zoom, rplot, azplot, (mx_r_zoom, mx_az_zoom), res_dict, r_vec, az_vec


def shift_array(array, shift):
    """
    Shift a multidimensional array and pads it with zeros.
    
    Parameters
    ----------
    array : ndarray
        the array to shift.
    shift  : tuple
        a tuple of shifts.
    Returns
    -------
    ndarray : 
        the shifted array
    """
    array_1 = array * 1
    pad_array = [(0, 0), ] * array.ndim
    for ax, current_shift in enumerate(shift):
        slice_array = [Ellipsis, ] * array.ndim
        if shift > 0:
            pad_array[ax] = (0, shift)
            slice_array[ax] = slice(shift)
        else:
            pad_array[ax] = (_np.abs(shift), 0)
            slice_array[ax] = slice(0, -shift)
    array_1_pad = _np.pad(array_1, tuple(pad_array), mode='const')
    array_1 = _np.roll(array_1_pad, current_shift, axis=ax)[tuple[slice_array]]
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
        if _np.nanmax(_np.abs(T_H - T).flatten()) < thresh:
            is_herm = True
        else:
            is_herm = False
        return is_herm
    except (TypeError, AttributeError):
        print("T is not a numpy array")


def transform(A, B, C):
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
    out = B.__array_wrap__(_np.einsum("...ik,...kl,...lj->...ij", A, B, C))
    return out


def complex_interpolate(image, osf):
    abs_interp = _nd.zoom(_np.abs(image), osf)
    ang_interp = _nd.zoom(_np.angle(image), osf)
    image_interp = abs_interp * _np.exp(1j * ang_interp)
    image_interp = image.__array_wrap__(image_interp)
    image_interp.GPRI_az_angle_step[0] = osf[1] * image.GPRI_az_angle_step[0]
    image_interp.range_pixel_spacing[0] = osf[0] * image.range_pixel_spacing[0]
    image_interp.azimuth_line_time[0] = osf[1] * image.azimuth_line_time[0]
    image_interp.prf[0] = osf[1] * image.prf[0]
    image_interp.azimuth_lines = image_interp.shape[1]
    image_interp.range_samples = image_interp.shape[0]
    return image_interp


def resample_geometry(slc, reference_slc):
    """
    This function resamples a daraset into the geometry of
    the other dataset by changing the azimuth & range spacing
    Parameters
    ----------
    slc
    reference_slc

    Returns
    -------

    """

    # TODO implement code to resample geometry of datasets
    r_osf = reference_slc.range_pixel_spacing[0] / slc.range_pixel_spacing[0]
    r_az = reference_slc.GPRI_az_angle_step[0] / slc.GPRI_az_angle_step[0]





def matrix_root(A):
    l, w = (_np.linalg.eig(_np.array(A)))
    l_sq = (_np.sqrt(l))
    if A.ndim > 2:
        L_sq = _np.zeros_like(w)
        for idx_diag in range(w.shape[2]):
            L_sq[:, :, idx_diag, idx_diag] = l_sq[:, :, idx_diag]
    else:
        L_sq = _np.diag(l_sq)
    A_sq = transform(w, L_sq, _np.linalg.inv(w))
    return A_sq


def window_idx(arr, idx, zs):
    indices = []
    for cnt, (current_size, current_idx) in enumerate(zip(zs, idx)):
        mi = _np.clip(current_idx - current_size // 2, 0, arr.shape[cnt])
        mx = _np.clip(current_idx + current_size // 2, 0, arr.shape[cnt])
        if mi == mx:
            sl = slice(mi, mx + 1)
        else:
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
    # Get indices to slice the array
    indices = window_idx(arr, idx, ws)
    # Slice the array
    arr_section = arr[indices]
    # Find the argmax
    max_idx = _np.argmax(arr_section)
    # Now that we have the indices, we convert it
    # to tuple
    max_idx = _np.unravel_index(max_idx, arr_section.shape)
    # We have now to convert it into the "coordinates" of the big array
    final_idx = [local_idx + global_idx - current_size //2 for
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


def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    # Compute padding for block
    pad = [block_dim - dim % block_dim for dim, block_dim in zip(A.shape, block)]
    pad = pad + (len(A.shape) - len(block)) * [0]
    total_pad = [(p, 0) for p in pad]
    A = _np.pad(A, total_pad, mode='constant')
    shape = (A.shape[0] / block[0], A.shape[1] / block[1]) + block + A.shape[len(block)::]
    strides = (block[0] * A.strides[0], block[1] * A.strides[1]) + A.strides
    return _ast(A, shape=shape, strides=strides)


def block_to_array(A, block=(3, 3)):
    new_shape = (A.shape[0] * block[0], A.shape[1] * block[1]) + A.shape[len(block) + 2::]
    strides = A.strides[len(block)::]
    return _ast(A, shape=new_shape, strides=strides)

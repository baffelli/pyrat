# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:56:18 2014

@author: baffelli
"""
import numpy as np

"""
Utilities for GPRI calibration
"""
import itertools as _itertools

import numpy as _np
from scipy import fftpack as _fftp, signal
from scipy import signal as _sig
from scipy import ndimage as _nd

from .. import core
from ..core import corefun
from ..fileutils import gpri_files as _gpf

import scipy.optimize as _opt

import matplotlib.pyplot as plt

import pyfftw.interfaces.scipy_fftpack as _fftp



import math as _math

def measure_phase_center_location(slc, ridx, azidx, sw=(2,10), aw=60, unwrap=True):

    #lever arm length is given by antenna phase center x position
    """
        Determine the antenna phase center shift according to the lever arm model
    Parameters
    ----------
    slc : fleutils.gammaDataset
        the gammaDataset object to analyze
    ridx : float
        range position of point target to analyzed
    azidx : float
        azimuth position of point target
    sw  : iterable
        search window in azimuth and range to determine the exact position
    aw  : float
        analysis window
    unwrap

    Returns
    -------

    """
    r_arm = slc.phase_center[0]
    lam = _gpf.C / slc.radar_frequency
    def cf(r_arm, r_ph, r, az_vec, off, meas_phase):
        sim_phase, dist = distance_from_phase_center(r_arm, r_ph, r, az_vec, lam, wrap=False)
        cost = _np.mean(_np.abs(sim_phase + off - meas_phase) ** 2)
        return cost
    # Find the maxium in search window
    max_r, max_az = corefun.maximum_around(_np.abs(slc), [ridx, azidx], sw)
    #Determine analysis window
    max_slice = corefun.window_idx(slc, (max_r, max_az), (1, aw))
    reflector_slice = slc[max_slice].squeeze()
    max_pwr = _np.max(_np.abs(reflector_slice))
    # Determine half power beamwidth
    half_pwr_idx = _np.nonzero(_np.abs(reflector_slice)**2 > max_pwr**2 * 0.5)
    # Slice slc
    reflector_slice = reflector_slice[half_pwr_idx]
    # Determine parameters
    r_vec = slc.r_vec
    az_vec = _np.deg2rad(slc.GPRI_az_angle_step) * _np.arange(-len(reflector_slice) / 2,
                                                                      len(reflector_slice) / 2)
    refl_ph = _np.angle(reflector_slice)
    if unwrap:
        refl_ph = _np.unwrap(refl_ph)
    else:
        refl_ph = refl_ph
    #reference the phase
    refl_ph -= refl_ph[refl_ph.shape[0] / 2.0]
    r_sl = r_vec[ridx]
    # Define cost function
    cost_VV = lambda par_vec: cf(r_arm, par_vec[0], r_vec[ridx], az_vec, par_vec[1], refl_ph)
    # Solve optimization problem
    res = _opt.minimize(cost_VV, [0, 0], bounds=((-2, 2), (None, None)), method="L-BFGS-B")
    print(res)
    par_dict = {'phase_center_offset': [res.x[0], 'm'], 'residual error': res.fun,
                'lever_arm_length': [r_arm, 'm'], 'range_of_closest_approach': [r_sl, 'm']}
    sim_ph, dist = distance_from_phase_center(r_arm, res.x[0], r_sl, az_vec, lam, wrap=False)
    sim_ph -= sim_ph[sim_ph.shape[0]/2]
    #Compute RVP
    rvp = _np.exp(1j * 4 * dist**2 * slc.chirp_bandwidth / _gpf.C**2)
    #Compute rcm
    rcm = _np.max(r_vec[ridx] - dist)
    return res.x[0], res.fun, r_sl, refl_ph, sim_ph, rvp, rcm


def range_resolution(B):
    return _gpf.C / (2 * B)

def calibrate_from_r_and_t(S, R, T):
    T_inv = _np.linalg.inv(T)
    R_inv = _np.linalg.inv(R)
    S_cal = corefun.transform(R_inv, S, T_inv)
    return S_cal

def cr_rcs(l, freq, type='triangular'):
    lam = _gpf.lam(freq)
    if type == 'triangular':
        RCS  = 4 / 3 * _np.pi * l**4 / (lam**2)
    elif type == 'cubic':
        RCS = 12 * _np.pi * l**4 / (lam**2)
    return RCS


def remove_window(S):
    spectrum = _np.mean(_np.abs((_fftp.fft(S, axis=1))), axis=0)
    spectrum = corefun.smooth(spectrum, 5)
    spectrum[spectrum < 1e-6] = 1
    S_f = _fftp.fft(S, axis=1)
    S_corr = _fftp.ifft(S_f / spectrum, axis=1)
    return S_corr



def symmetric_pad(pad_size):
    # if pad_size % 2 == 0:
    #     left_pad = right_pad = np.ceil(-pad_size//2)  + 1
    # else:
    left_pad = int(np.floor(pad_size/2)) + 1
    right_pad = int(np.ceil(pad_size/2))
    pad_vec = (left_pad, right_pad)
    return pad_vec

def filter2d(slc, filter):
    filter_pad_size = ((0,0), symmetric_pad(slc.shape[1]+filter.shape[1]))
    filter_pad = np.pad(filter, filter_pad_size, mode='constant')
    slc_pad_size = ((0,0,), symmetric_pad(filter.shape[1]*2))
    slc_pad = np.pad(slc,slc_pad_size, mode='constant')
    #Transform
    fft_fun = lambda arr: _fftp.fft(arr, axis=1)
    ifft_funt = lambda arr: _fftp.ifftshift(_fftp.ifft(arr, axis=1),axes=(1,))
    filter_hat = fft_fun(filter_pad)
    slc_hat = fft_fun(slc_pad)
    #Product and invese transform
    slc_filt = ifft_funt(slc_hat * filter_hat)
    slc_filt = slc_filt[:, slice(slc_pad_size[1][0],slc_filt.shape[1]-slc_pad_size[1][1])]
    return slc_filt

def filter1d(slc, filter):
    slc_filt = slc * 0
    for idx_row, (current_row, current_filter) in enumerate(_itertools.zip_longest(slc, filter)):
        if idx_row % 1000 == 0:
            print('Processing range index: ' + str(idx_row))
        # _sig.fftconvolve(current_row.real, current_filter.real, mode='same') + 1j * _sig.fftconvolve(current_row.imag,
        #                                                                                              current_filter.imag,
        #                                                                                              mode='same')
        slc_filt[idx_row,:] = _sig.convolve(current_row, current_filter, mode='same')
    return slc_filt


def azimuth_correction(slc, r_ph, ws=0.6, discard_samples=False, filter_fun=filter1d):
    r_arm = _np.linalg.norm(slc.phase_center[0:2])
    # Azimuth vector for the entire image
    # az_vec_image = _np.deg2rad(self.slc.GPRI_az_start_angle[0]) + _np.arange(self.slc.shape[0]) * _np.deg2rad(
    #     self.slc.GPRI_az_angle_step[0])
    # Compute integration window size in samples
    ws_samp = ws // slc.GPRI_az_angle_step
    # Filtered slc has different sizes depending
    # if we keep all the samples after filtering
    # if not discard_samples:
    #     slc_filt = slc * 1
    # else:
    #     slc_filt = slc[:, ::ws_samp] * 1
    # process each range line
    theta = _np.arange(-ws_samp // 2, ws_samp // 2) * _np.deg2rad(slc.GPRI_az_angle_step)
    rr, tt = np.meshgrid(slc.r_vec, theta, indexing='ij')
    lam = _gpf.C / slc.radar_frequency
    filt2d, dist2d = distance_from_phase_center(r_arm, r_ph, rr, tt, lam, wrap=False)
    matched_filter2d = (_np.exp(-1j * filt2d))
    #Convert to fourier domain
    slc_filt_2d = filter_fun(slc.astype(np.complex64), matched_filter2d)
    slc_filt = slc.__array_wrap__(slc_filt_2d).astype(slc.dtype)
    if discard_samples:
        slc_filt = slc_filt.decimate(ws_samp, mode='discard')
    return slc_filt


def measure_imbalance(C_tri, C):
    """
    This function measures the co and crosspolar imbalance
    Parameters
    ----------
    C_tri   : coherencyMatrix
        the covariance image from which to estimate the parameters (must be averaged)
    C   : coherencyMatrix
    the covariance to estimate the HV VH imbalanace
    Returns
    -------

    """
    f = (_np.abs(C_tri[3, 3]) / _np.abs(C_tri[0, 0])) ** (1 / 4.0)
    VV_HH_phase_bias = _np.angle(C_tri[3, 0])
    g = (_np.mean(C[:, :, 1, 1]) / _np.mean(C[:, :, 2, 2])) ** (1 / 4.0)
    cross_pol_bias = _np.angle(_np.mean((C[:, :, 1, 2])))
    # Solve for phi t and phi r
    phi_t = (VV_HH_phase_bias + cross_pol_bias) / 2
    phi_r = (VV_HH_phase_bias - cross_pol_bias) / 2
    # A =  (10**(rcs/10) / (_np.abs(C_tri[0,0])))**(1/4.0)
    return phi_t, phi_r, f, g


def gpri_radcal(mli, tri_pos, sigma):
    """
    Compute the GPRI radiometric calibration parameter
    Parameters
    ----------
    mli : gammaDataset
        the MLI image to use to determine the calibratio constant
    tri_pos : iterable
        the location of the calibration area
    sigma : float


    Returns
    -------

    """
    # extract the point target response to get the maximum
    mli_ptarg, rplot, azplot, mx_idx, res_dic, r_vec, az_vec = corefun.ptarg(mli, tri_pos[0], tri_pos[1], sw=(2,2), rwin=5, azwin=5)
    # illuminated area
    A_illum = mli.range_pixel_spacing *_np.deg2rad(0.4) * mli.r_vec[tri_pos[0]]
    # Calibration factor
    K = sigma/(mli_ptarg[mx_idx] * A_illum)
    return K


def distortion_matrix(phi_t, phi_r, f, g):
    dm = _np.diag([1, f * g * _np.exp(1j * phi_t), f / g * _np.exp(1j * phi_r),
                                  f ** 2 * _np.exp(1j * (phi_r + phi_t))])
    return dm


def calibrate_from_parameters(S, par):
    # TODO check calibration
    """
    This function performs polarimetric calibration from a text file containing the m-matrix distortion parameters
    Parameters
    ----------
    S : scatteringMatrix, coherencyMatrix
        the matrix to calibrate
    par : string, array_like
        Path to the parameters file, it must in the form of _np.save
    """
    if isinstance(par, str):
        m = _np.fromfile(par)
        m = m[::2] + m[1::2]
    elif isinstance(par, (_np.ndarray, list, tuple)):
        m = par
    else:
        raise TypeError("The parameters must be passed\
        as a list or array or tuple or either as a path to a text file")
    if isinstance(S, core.matrices.scatteringMatrix):
        f = m[0]
        g = m[1]
        S_cal = S.__copy__()
        S_cal['VV'] = S_cal['VV'] * 1 / f ** 2
        S_cal['HV'] = S_cal['HV'] * 1 / f * 1 / g
        S_cal['VH'] = S_cal['VH'] * 1 / f
        S_cal = S.__array_wrap__(S_cal)
    return S_cal


def scattering_matrix_to_flat_covariance(S, flat_ifgram, B_if):
    C = S.to_coherency_matrix(basis='lexicographic', bistatic=True)
    # Convert mapping into list of tuples
    mapping_list = [(key, value) for key, value in _gpf.channel_dict.items()]
    for (name_chan_1, idx_chan_1), (name_chan_2, idx_chan_2) in _itertools.product(mapping_list, mapping_list):
        # convert the indices of the two channels into the indices to access the covariance component
        idx_c_1 = _np.ravel_multi_index(idx_chan_1, (2, 2))
        idx_c_2 = _np.ravel_multi_index(idx_chan_2, (2, 2))
        pol_baseline = S.phase_center_array[name_chan_1][-1] - S.phase_center_array[name_chan_2][-1]
        ratio = pol_baseline / float(B_if)
        C[:, :, idx_c_1, idx_c_2] *= _np.exp(-1j * flat_ifgram * ratio)
    return C


def coregister_channels(S):
    """
    This function coregisters the GPRI channels by shifting each channel by the corresponding number of samples in azimuth
    It assumes the standard quadpol AAA-ABB-BAA-BBB TX-RX-seq.
    
    Parameters
    ----------
    S : pyrat.core.matrices.scatteringMatrix
        The scattering matrix to be coregistered.
    
    Returns
    -------
    scatteringMatrix
        The image after coregistration.
    """
    S1 = S
    S1['VV'] = corefun.shift_array(S['VV'], (3, 0))
    S1['HV'] = corefun.shift_array(S['HV'], (1, 0))
    S1['VH'] = corefun.shift_array(S['VH'], (2, 0))
    return S1



def remove_window(S):
    spectrum = _np.mean(_np.abs(_fftp.fftshift(_fftp.fft(S, axis=1), axes=(1,))), axis=0)
    spectrum = corefun.smooth(spectrum, 5)
    spectrum[spectrum < 1e-6] = 1
    S_f = _fftp.fft(S, axis=1)
    S_corr = _fftp.ifft(S_f / _fftp.fftshift(spectrum), axis=1)
    return S_corr


def synthetic_interferogram(S, DEM, B):
    """
    Parameters
    ----------
    S : scatteringMatrixc
        The image to correct
    DEM : ndarray
        A DEM in the same coordinates as the image
    """

    # Coordinate System W.R.T antenna
    r, th = _np.meshgrid(S.r_vec, S.az_vec)
    x = r * _np.cos(th)
    y = r * _np.sin(th)
    z = DEM
    r1 = _np.dstack((x, y, z))
    r2 = _np.dstack((x, y, (z + B)))
    # Convert into
    delta_d = _np.linalg.norm(r1, axis=2) - _np.linalg.norm(r2, axis=2)
    lamb = 3e8 / S.center_frequency
    ph = delta_d * 4 / lamb * _np.pi
    return _np.exp(1j * ph)



def distance_from_phase_center(L_arm, L_ph, R_0, theta, lam, wrap=False):
    """
    This function computes the relative phase caused by a shifted
    phase center in the antenna. The variation is computed relative to the slante of closest approach
    """
    # lam = _gpf.C / 17.2e9
    L_ant = _np.sqrt(L_arm ** 2 + L_ph ** 2)
    alpha = _np.arctan(L_ph / L_arm)
    # Chord length
    c = L_ant + R_0
    mixed_term = 2 * c * L_ant * _np.cos(theta + alpha)
    dist =  _np.sqrt(c ** 2 + L_ant ** 2 - mixed_term)
    rel_dist = R_0 - dist
    if wrap is True:
        return _np.mod(4 * _np.pi * rel_dist / lam, 2 * _np.pi), dist
    else:
        return (4 * _np.pi * rel_dist / lam), dist


def squint_vec(rawdata, z=2500):
    win_slice = slice(z, rawdata.shape[0] - z)
    rawdata_sl = rawdata[win_slice,:]#window the edges
    max_idx = np.argmax(np.abs(rawdata_sl), axis=1)
    return max_idx, win_slice


def fit_squint(raw, slc_par, azidx, ridx, win=(10,10), z=2500):
    """
    Performs a fit of squint-angle verus chirp frequency on the raw data
    by analyzing the response of a point-like target
    Parameters
    ----------
    raw : pyrat.fileutils.rawData
    slc_par : pyrat.fileutils.ParameterFile
    azidx :  int
        index of point target in azimuth
    ridx : int
        index of point target in range
    win : tuple
        window to extract around the point target
    z : integer
        number to samples to discard and the beginning and end of each chirp

    Returns
    -------

    """
    az_slice = raw.azimuth_slice_from_slc_idx(azidx, win[1])
    # Construct
    raw_sl = raw[:, az_slice] * 1
    # Range filter
    raw_filt = _sig.hilbert(raw_sl.filter_range_spectrum(slc_par, ridx, win[0], k=2), axis=0)
    az_vec = np.arange(-raw_filt.shape[1] // 2, raw_filt.shape[1] // 2) * raw.azspacing
    # Find maximum
    squint_idx, win_slice = squint_vec(raw_filt, z=z)
    squint = az_vec[squint_idx[::-1]]
    # fit squint
    # w = np.abs(np.array([row[squint_idx[idx]] for idx,row in enumerate(raw_filt[win_slice])]))
    sq_par = np.polyfit(raw.freqvec[win_slice], squint, 1)
    return squint_idx, squint, sq_par, raw_filt
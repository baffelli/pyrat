# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:56:18 2014

@author: baffelli
"""

"""
Utilities for GPRI calibration
"""
import itertools as _itertools

import numpy as _np
from scipy import fftpack as _fftp
from scipy import signal as _sig
from scipy import ndimage as _nd

from .. import core
from ..core import corefun
from ..fileutils import gpri_files as _gpf

import scipy.optimize as _opt

import matplotlib.pyplot as plt

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

    def cf(r_arm, r_ph, r, az_vec, off, meas_phase):
        sim_phase, dist = distance_from_phase_center(r_arm, r_ph, r, az_vec, wrap=False)
        cost = _np.mean(_np.abs(sim_phase + off - meas_phase) ** 2)
        return cost
    # Find the maxium in search window
    max_r, max_az = corefun.maximum_around(_np.abs(slc), [ridx, azidx], sw)
    #Determine analysis window
    reflector_slice = slc[max_r, slice(max_az - aw/2, max_az + aw/2)]
    max_pwr = _np.max(_np.abs(reflector_slice))
    # Determine half power beamwidth
    half_pwr_idx = _np.nonzero(_np.abs(reflector_slice)**2 > max_pwr**2 * 0.5)
    # Slice slc
    reflector_slice = reflector_slice[half_pwr_idx]
    # Determine parameters
    r_vec = slc.r_vec
    az_vec = _np.deg2rad(slc.GPRI_az_angle_step[0]) * _np.arange(-len(reflector_slice) / 2,
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
    sim_ph, dist = distance_from_phase_center(r_arm, res.x[0], r_sl, az_vec, wrap=False)
    sim_ph -= sim_ph[sim_ph.shape[0]/2]
    return res.x[0], res.fun, r_sl, refl_ph, sim_ph


def calibrate_from_r_and_t(S, R, T):
    T_inv = _np.linalg.inv(T)
    R_inv = _np.linalg.inv(R)
    S_cal = corefun.transform(R_inv, S, T_inv)
    return S_cal


def remove_window(S):
    spectrum = _np.mean(_np.abs((_fftp.fft(S, axis=1))), axis=0)
    spectrum = corefun.smooth(spectrum, 5)
    spectrum[spectrum < 1e-6] = 1
    S_f = _fftp.fft(S, axis=1)
    S_corr = _fftp.ifft(S_f / spectrum, axis=1)
    return S_corr


def azimuth_correction(slc, r_ph, ws=0.6, discard_samples=False):
    r_ant = _np.linalg.norm(slc.phase_center[0:2])
    print(r_ant)
    # Construct range vector
    # r_vec = self.slc.near_range_slc[0] + _np.arange(self.slc.shape[0]) * self.slc.range_pixel_spacing[0]
    r_vec = slc.r_vec
    # Azimuth vector for the entire image
    # az_vec_image = _np.deg2rad(self.slc.GPRI_az_start_angle[0]) + _np.arange(self.slc.shape[0]) * _np.deg2rad(
    #     self.slc.GPRI_az_angle_step[0])
    # Compute integration window size in samples
    ws_samp = ws // slc.GPRI_az_angle_step[0]
    # Filtered slc has different sizes depending
    # if we keep all the samples after filtering
    if not discard_samples:
        slc_filt = slc * 1
    else:
        slc_filt = slc[:, ::ws_samp] * 1
    # process each range line
    theta = _np.arange(-ws_samp // 2, ws_samp // 2) * _np.deg2rad(slc.GPRI_az_angle_step[0])
    for idx_r, r_sl in enumerate(r_vec):
        filt, dist = distance_from_phase_center(r_ant, r_ph, r_sl, theta, wrap=False)
        lam = _gpf.C / 17.2e9
        # Normal matched filter
        matched_filter = _np.exp(-1j * filt) * _np.exp(-1j * 4 * _np.pi * r_sl / lam)
        filter_output = _sig.convolve(slc[idx_r, :], matched_filter, mode='same')
        if discard_samples:
            filter_output = filter_output[::ws_samp]
            slc_filt.GPRI_az_angle_step[0] = slc.GPRI_az_angle_step[0] * ws_samp
            slc_filt.azimuth_lines = filter_output.shape[0]
        else:
            pass
        slc_filt[idx_r, :] = filter_output
        slc_filt = slc.__array_wrap__(slc_filt)
        if idx_r % 1000 == 0:
            print('Processing range index: ' + str(idx_r))
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
    mli_ptarg, rplot, azplot, mx_idx, res_dic = corefun.ptarg(mli, tri_pos[0], tri_pos[1])
    # illuminated area
    A_illum = mli.range_pixel_spacing[0] *_np.deg2rad(0.4) * mli.r_vec[tri_pos[0]]
    # Calibration factor
    K = sigma/(mli_ptarg[mx_idx])
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


# def coregister_channels_FFT(S, shift_patch, oversampling=(5, 5)):
#     """
#     This function coregisters the GPRI channels by shifting each channel by the corresponding number of samples in azimuth. The shifting is computed
#     using the cross-correlation on a specified patch.
#     It assumes the standard quadpol AAA-ABB-BAA-BBB TX-RX-seq
#     -----
#     Parameters
#     S : scatteringMatrix
#     The scattering matrix to be coregistered
#     shift_patch : tuple
#         slice indices of the patch where to perform the coregistration
#         works best if the patch contains a single bright object such as corner reflector
#     oversampling : int
#     the oversampling factor for the FFT
#     -----
#     Returns
#     scatteringMatrix
#         The image after coregistration
#     """
#     S_cor = S * 1
#     co_shift, corr_co = get_shift(_np.abs(S['HH'][shift_patch]), _np.abs(S['VV'][shift_patch]), axes=(0, 1),
#                                   oversampling=oversampling)
#     cross_shift, cross_co = get_shift(_np.abs(S['HH'][shift_patch]), _np.abs(S['HV'][shift_patch]), axes=(0, 1),
#                                       oversampling=oversampling)
#     cross_shift_1, cross_co = get_shift(_np.abs(S['HH'][shift_patch]), _np.abs(S['VH'][shift_patch]), axes=(0, 1),
#                                         oversampling=oversampling)
#     S_cor['VV'] = _vf.shift_image(S['VV'], co_shift)
#     S_cor['HV'] = _vf.shift_image(S['HV'], cross_shift)
#     S_cor['VH'] = _vf.shift_image(S['VH'], cross_shift_1)
#     return S_cor


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


# def get_shift(image1, image2, oversampling=(10, 1), axes=(0, 1)):
#     corr_image = norm_xcorr(image1, image2, axes=axes, oversampling=oversampling)
#     # Find the maximum index
#     shift = _np.argmax(_np.abs(corr_image))
#     # Unroll it to have the 2D indices
#     shift = _np.unravel_index(shift, corr_image.shape)
#     # if shift is larger than half array
#     # we have a negative shift
#     half_shape = (_np.array(corr_image.shape) / 2).astype(_np.int)
#     pos_shift = half_shape - _np.array(shift)
#     neg_shift = -1 * (half_shape - _np.array(shift))
#     shift_idx = _np.where(_np.array(shift) > (_np.array(corr_image.shape) / 2.0)
#                           , pos_shift, neg_shift)
#     shift_idx = shift_idx / _np.array(oversampling).astype(_np.double)
#     return shift_idx, corr_image


# def norm_xcorr(image1, image2, axes=(0, 1), oversampling=(2, 2), pad_factor=(0.5, 0.5)):
#     import pyfftw.interfaces.scipy_fftpack as fftp
#     # Remove nans
#     image1[_np.isnan(image1)] = 0
#     image2[_np.isnan(image2)] = 0
#     # Pad edges to reduce edge effects
#     edge_pad_size = zip([0] * image1.ndim, [0] * image1.ndim)
#     for ax, pf in zip(axes, pad_factor):
#         ps = image1.shape[ax] * (pf / 2)
#         edge_pad_size[ax] = (ps, ps)
#     edge_pad_size = tuple(edge_pad_size)
#     #    image1 = _np.pad(image1, edge_pad_size, mode='constant')
#     #    image2 = _np.pad(image2, edge_pad_size, mode='constant')
#     # Take trnasform
#     image_1_hat = fftp.fftn(image1, axes=axes)
#     image_2_hat = fftp.fftn(image2, axes=axes)
#     # Oversample
#     pad_size = zip([0] * image1.ndim, [0] * image1.ndim)
#     for ax, ov in zip(axes, oversampling):
#         os = image1.shape[ax] * (ov - 1)
#         pad_size[ax] = (os / 2, os / 2)
#     pad_size = tuple(pad_size)
#     ft_corr = image_1_hat * image_2_hat.conj() \
#               / (_np.abs(image_1_hat * image_2_hat.conj()))
#     ft_corr_pad = _np.pad(ft_corr, pad_size, mode='constant')
#     phase_corr = fftp.fftshift(fftp.ifftn(ft_corr_pad, axes=axes) \
#                                , axes=axes)
#     return phase_corr
#
#
# def patch_coregistration(im1, im2, n_patch, oversampling=(5, 5)):
#     rem = _np.array(n_patch) - _np.mod(im1.shape, n_patch)
#     pad_size = [0] * im1.ndim
#     for ax in range(im1.ndim):
#         ps = rem[ax]
#         pad_size[ax] = (ps, 0)
#     im1 = _np.pad(im1, pad_size, mode='constant')
#     im2 = _np.pad(im2, pad_size, mode='constant')
#     sh = _np.divide(im1.shape, n_patch)
#     patches_1 = matrices.blockshaped(im1, sh[0], sh[1])
#     patches_2 = matrices.blockshaped(im2, sh[0], sh[1])
#     sh_list = []
#     for p1, p2 in zip(patches_1, patches_2):
#         block_idxs = _np.unravel_index(idx_block, patches_1.shape[0:2])
#         sh, co = get_shift(p1, p2, oversampling=oversampling)
#         #        sh_arr[block_idxs] = sh[0] + 1j *sh[1]
#         sh_list.append(sh)
#     return sh_list


def distance_from_phase_center(r_arm, r_ph, r_sl, theta, wrap=False):
    """
    This function computes the phase caused by a shifted
    phase center in the antenna
    """
    lam = _gpf.C / 17.2e9
    r_ant = _np.sqrt(r_arm ** 2 + r_ph ** 2)
    alpha = _np.arctan(r_ph / r_arm)
    # Chord length
    c = r_ant + r_sl
    mixed_term = 2 * c * r_ant * _np.cos(theta + alpha)
    dist = _np.sqrt(c ** 2 + r_ant ** 2 - mixed_term)
    if wrap is True:
        return _np.mod(-4 * _np.pi * dist / lam, 2 * _np.pi), dist
    else:
        return (-4 * _np.pi * dist / lam), dist

# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:56:18 2014

@author: baffelli
"""

"""
Utilities for GPRI calibration
"""
import numpy as _np
from .. import core
from .. import fileutils as gpf
from ..core import corefun, polfun
from ..visualization import visfun as _vf
from scipy import fftpack as _fftp
from ..fileutils import gpri_files as _gpf
from scipy import signal as _sig

def calibrate_from_r_and_t(S, R,T):
    T_inv = _np.linalg.inv(T)
    R_inv = _np.linalg.inv(R)
    S_cal = corefun.transform(R_inv,S,T_inv)
    return S_cal


def remove_window(S):
    spectrum = _np.mean(_np.abs((_fftp.fft(S,axis = 1))),axis = 0)
    spectrum = corefun.smooth(spectrum,5)
    spectrum[spectrum < 1e-6] = 1
    S_f = _fftp.fft(S,axis = 1)
    S_corr = _fftp.ifft(S_f / (spectrum), axis = 1)
    return S_corr


def azimuth_correction(slc, r_ph, ws=0.4, discard_samples=False):
    r_ant = _gpf.xoff + _np.cos(_np.deg2rad(slc.GPRI_ant_elev_angle[0])) * _gpf.ant_radius
    # Construct range vector
    # r_vec = self.slc.near_range_slc[0] + _np.arange(self.slc.shape[0]) * self.slc.range_pixel_spacing[0]
    r_vec = slc.r_vec
    # Azimuth vector for the entire image
    # az_vec_image = _np.deg2rad(self.slc.GPRI_az_start_angle[0]) + _np.arange(self.slc.shape[0]) * _np.deg2rad(
    #     self.slc.GPRI_az_angle_step[0])
    az_vec = slc.az_vec
    # Compute integration window size in samples
    ws_samp = int(ws / slc.GPRI_az_angle_step[0])
    # Filtered slc has different sizes depending
    # if we keep all the samples after filtering
    if not discard_samples:
        slc_filt = slc * 1
    else:
        slc_filt = slc[:, ::ws_samp] * 1
    # process each range line
    theta = _np.arange(-ws_samp / 2, ws_samp / 2) * _np.deg2rad(slc.GPRI_az_angle_step[0])
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
    refpos  : iterable
        the position of the TCR (ridx,azidx)
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
    return phi_t, phi_r, f, g

def distortion_matrix(phi_t, phi_r, f, g):
    distortion_matrix = _np.diag([1, f * g * _np.exp(1j * phi_t), f/g * _np.exp(1j * phi_r),
                                  f**2 * _np.exp(1j * (phi_r + phi_t))])
    return distortion_matrix

def calibrate_from_parameters(S,par):
    #TODO check calibration
    """
    This function performs polarimetric calibration from a text file containing the m-matrix distortion parameters
    Parameters
    ----------
    S : scatteringMatrix, coherencyMatrix
        the matrix to calibrate
    par : string, array_like
        Path to the parameters file, it must in the form of _np.save
    """
    if  isinstance(par,str):
        m = _np.fromfile(par)
        m = m[::2] + m[1::2]
    elif isinstance(par,(_np.ndarray,list,tuple)):
        m = par
    else:
        raise TypeError("The parameters must be passed\
        as a list or array or tuple or either as a path to a text file")
    if isinstance(S, core.matrices.scatteringMatrix):
        f = m[0]
        g = m[1]
        S_cal = S.__copy__()
        S_cal['VV'] = S_cal['VV'] * 1/f**2 
        S_cal['HV'] = S_cal['HV'] * 1/f * 1/g
        S_cal['VH'] = S_cal['VH'] * 1/f
        S_cal = S.__array_wrap__(S_cal)
    return S_cal




def remove_phase_ramp(S1, B_if, ph_if, bistatic = False, S2 = None):
    if S2 is None:
        S2 = S1 * 1
    else:
        pass
    k1 = S1.scattering_vector(basis = 'lexicographic', bistatic = bistatic)
    k2 = S2.scattering_vector(basis = 'lexicographic', bistatic = bistatic)
    C = corefun.outer_product(k1, k2)
    dummy = S1.to_coherency_matrix(basis = 'lexicographic')
    corr_mat = _np.zeros(C.shape,dtype = _np.complex64)
    if bistatic is True:
        channel_vec = ['HH','HV','VH','VV']
    else:
        channel_vec = ['HH','HV','VV']
    for idx_1 in range(C.shape[-1]):
        for idx_2 in range(C.shape[-1]):
            pol_baseline = S1.ant_vec[channel_vec[idx_1]] - S2.ant_vec[channel_vec[idx_2]]
            cf = (pol_baseline) / B_if
            corr = _np.exp(1j*ph_if * cf)
            corr_mat[:, :,idx_1,idx_2] = corr
    C_cal = C * corr_mat
    C_cal = dummy.__array_wrap__(C_cal)
    return C_cal




def coregister_channels(S):
    """
    This function coregisters the GPRI channels by shifting each channel by the corresponding number of samples in azimuth
    It assumes the standard quadpol AAA-ABB-BAA-BBB TX-RX-seq.
    
    Parameters
    ----------
    S : scatteringMatrix
        The scattering matrix to be coregistered.
    
    Returns
    -------
    scatteringMatrix
        The image after coregistration.
    """
    S1 = S * 1
    S1['VV'] = corefun.shift_array(S['VV'],(3,0))
    S1['HV'] = corefun.shift_array(S['HV'],(1,0))
    S1['VH'] = corefun.shift_array(S['VH'],(2,0))
    return S1

def coregister_channels_FFT(S,shift_patch, oversampling = (5,5)):
    """
    This function coregisters the GPRI channels by shifting each channel by the corresponding number of samples in azimuth. The shifting is computed
    using the cross-correlation on a specified patch.
    It assumes the standard quadpol AAA-ABB-BAA-BBB TX-RX-seq
    -----
    Parameters
    S : scatteringMatrix
    The scattering matrix to be coregistered
    shift_patch : tuple
        slice indices of the patch where to perform the coregistration
        works best if the patch contains a single bright object such as corner reflector
    oversampling : int
    the oversampling factor for the FFT
    -----
    Returns
    scatteringMatrix
        The image after coregistration
    """
    S_cor = S * 1
    co_shift, corr_co = get_shift(_np.abs(S['HH'][shift_patch]),_np.abs(S['VV'][shift_patch]), axes = (0,1),
                                  oversampling = oversampling)
    cross_shift, cross_co = get_shift(_np.abs(S['HH'][shift_patch]),_np.abs(S['HV'][shift_patch]), axes = (0,1),
                                      oversampling = oversampling)
    cross_shift_1, cross_co = get_shift(_np.abs(S['HH'][shift_patch]),_np.abs(S['VH'][shift_patch]), axes = (0,1), 
                                                oversampling = oversampling)
    S_cor['VV'] = _vf.shift_image(S['VV'],co_shift)
    S_cor['HV'] = _vf.shift_image(S['HV'],cross_shift)
    S_cor['VH'] = _vf.shift_image(S['VH'],cross_shift_1)
    return S_cor




        
def remove_window(S):
    spectrum = _np.mean(_np.abs(_fftp.fftshift(_fftp.fft(S,axis = 1),axes = (1,))),axis = 0)
    spectrum = core.corefun.smooth(spectrum,5)
    spectrum[spectrum < 1e-6] = 1
    S_f = _fftp.fft(S,axis = 1)
    S_corr = _fftp.ifft(S_f / _fftp.fftshift(spectrum), axis = 1)
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
    
    #Coordinate System W.R.T antenna
    r, th = _np.meshgrid(S.r_vec , S.az_vec)
    x =  r * _np.cos(th)
    y =  r * _np.sin(th)
    z =  DEM
    r1 = _np.dstack(( x,  y,  z))
    r2 = _np.dstack((x,  y, (z + B)))
    #Convert into 
    delta_d = _np.linalg.norm(r1, axis =2) - _np.linalg.norm(r2,axis = 2)
    lamb = 3e8/S.center_frequency
    ph = delta_d * 4 / lamb * _np.pi 
    return _np.exp(1j *ph)




    



    




    
def get_shift(image1,image2, oversampling = (10,1), axes = (0,1)):
    corr_image = norm_xcorr(image1, image2, axes = axes, oversampling = oversampling)
    #Find the maximum index
    shift = _np.argmax(_np.abs(corr_image))
    #Unroll it to have the 2D indices
    shift = _np.unravel_index(shift, corr_image.shape) 
    #if shift is larger than half array
    #we have a negative shift
    half_shape = (_np.array(corr_image.shape) / 2).astype(_np.int)
    pos_shift =   half_shape - _np.array(shift) 
    neg_shift =  -1 * ( half_shape - _np.array(shift)  )
    shift_idx = _np.where(_np.array(shift) > (_np.array(corr_image.shape) / 2.0)
    ,pos_shift, neg_shift)
    shift_idx = shift_idx  / _np.array(oversampling).astype(_np.double)
    return shift_idx, corr_image
    
    
def norm_xcorr(image1,image2, axes =(0,1),  oversampling =(2,2), pad_factor=(0.5,0.5)):
    import pyfftw.interfaces.scipy_fftpack as fftp
    #Remove nans
    image1[_np.isnan(image1)] = 0
    image2[_np.isnan(image2)] = 0
    #Pad edges to reduce edge effects
    edge_pad_size = zip([0] * image1.ndim,[0] * image1.ndim)
    for ax, pf in zip(axes, pad_factor):
        ps = image1.shape[ax] * (pf/2)
        edge_pad_size[ax] = (ps , ps)
    edge_pad_size = tuple(edge_pad_size)
#    image1 = _np.pad(image1, edge_pad_size, mode='constant')
#    image2 = _np.pad(image2, edge_pad_size, mode='constant')
    #Take trnasform
    image_1_hat = fftp.fftn(image1, axes = axes)
    image_2_hat = fftp.fftn(image2, axes = axes)
    #Oversample
    pad_size = zip([0] * image1.ndim,[0] * image1.ndim)
    for ax, ov in zip(axes, oversampling):
        os = image1.shape[ax] * (ov - 1)
        pad_size[ax] = (os/2, os/2)
    pad_size = tuple(pad_size)
    ft_corr = image_1_hat * image_2_hat.conj()\
    / (_np.abs( image_1_hat * image_2_hat.conj()))
    ft_corr_pad = _np.pad(ft_corr, pad_size, mode='constant')
    phase_corr = fftp.fftshift(fftp.ifftn(ft_corr_pad,axes = axes)\
    ,axes = axes)
    return phase_corr


def patch_coregistration(im1, im2, n_patch, oversampling = (5,5)):
    rem = _np.array(n_patch) - _np.mod(im1.shape, n_patch)
    pad_size = [0] * im1.ndim
    for ax in range(im1.ndim):
        ps = rem[ax]
        pad_size[ax] = (ps, 0)
    im1 = _np.pad(im1, pad_size, mode='constant')
    im2 = _np.pad(im2, pad_size, mode='constant')
    sh = _np.divide(im1.shape, n_patch)
    patches_1 = matrices.blockshaped(im1, sh[0], sh[1])
    patches_2 = matrices.blockshaped(im2, sh[0], sh[1])
    sh_list = []
    for p1, p2 in zip(patches_1, patches_2):
        block_idxs = _np.unravel_index(idx_block, patches_1.shape[0:2])
        sh, co = get_shift(p1, p2, oversampling = oversampling)
#        sh_arr[block_idxs] = sh[0] + 1j *sh[1]
        sh_list.append(sh)
    return sh_list


def distance_from_phase_center(r_arm, r_ph, r_sl, theta, wrap=False):
    """
    This function computes the phase caused by a shifted
    phase center in the antenna
    """
    lam = gpf.gpri_files.C / 17.2e9
    r_ant = _np.sqrt(r_arm**2 + r_ph**2)
    alpha = _np.arctan(r_ph / r_arm)
    #Chord length
    c = r_ant + r_sl
    mixed_term = 2 * c * r_ant * _np.cos(theta + alpha)
    dist = _np.sqrt(c**2 + r_ant**2 - mixed_term)
    if wrap is True :
        return _np.mod(-4 * _np.pi * dist/lam, 2 * _np.pi), dist
    else:
        return (-4 * _np.pi * dist/lam), dist

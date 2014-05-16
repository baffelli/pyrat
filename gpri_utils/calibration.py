# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:56:18 2014

@author: baffelli
"""

"""
Utilities for GPRI calibration
"""
import numpy as np
from ..core import corefun
#from ..core import scatteringMatrix

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
    S1['HH'] = corefun.shift_array(S['HH'],(-3,0))
    S1['HV'] = corefun.shift_array(S['HV'],(-2,0))
    S1['HV'] = corefun.shift_array(S['VH'],(-1,0))
    return S1

def coregister_channels_FFT(S,shift_patch, oversampling = 1):
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
    co_shift, corr_co = corefun.get_shift(np.abs(S['HH'][shift_patch]),np.abs(S['VV'][shift_patch]), axes = (0,1), oversampling = 1)
    cross_shift, cross_co = corefun.get_shift(np.abs(S['HH'][shift_patch]),np.abs(S['HV'][shift_patch]), axes = (0,1), oversampling = 1)
    cross_shift_1, cross_co = corefun.get_shift(np.abs(S['HH'][shift_patch]),np.abs(S['VH'][shift_patch]), axes = (0,1), oversampling = 1)
    S_cor['VV'] = corefun.shift_array(S['VV'],(co_shift[0],0))
    S_cor['HV'] = corefun.shift_array(S['HV'],(cross_shift[0],0))
    S_cor['VH'] = corefun.shift_array(S['VH'],(cross_shift_1[0],0))
    return S_cor


def correct_phase_ramp(S,if_phase, conversion_factor = 1):
    """ 
    This function corrects the interferometric phase ramp between the copolarized channels due to the spatial separation
    of the H and the V antenna. In order to do so, it requires a interferogram and the conversion factor between the polarimetric baseline 
    and the interferometric baseline
    
    Parameters    
    ----------
    S : scatteringMatrix
        `S` is the scattering matrix to be corrected
    if_phase : ndarray
        `if_phase` is interferometric phase to be subtracted
    conversion_factor : double or int, optional
        `conversion_factor` is the ratio of the normal baselines, used to convert the phase from one baseline to the other
    Returns
    -------
    S_corr : scatteringMatrix
        `S_corr` is the scattering matrix with the removed interferometric phase
    """
    S_corr = S * 1
    conversion_factor = 1
    S_corr['VV'] = S['VV'] * np.exp(-1j*conversion_factor*if_phase)
    return S_corr
    
def correct_phase_ramp_GPRI(S,S_other):
    """ 
    This function corrects the interferometric phase ramp between the copolarized channels due to the spatial separation
    of the H and the V antenna. To do so, it uses the interferometric baseline of the GPRI.
    Parameters
    ----------
    S : scatteringMatrix 
        the scattering matrix to be corrected
    S_other : scatteringMatrix
        An image taken at the other end of the baseline
    -----
    Returns
    scatteringMatrix
        the scattering matrix with the removed interferometric phase
    """
    HH_VV_if = np.angle(S['HH']*S_other['HH'].conj())
    S_corr = correct_phase_ramp(S,HH_VV_if)
    return S_corr

def correct_absolute_phase(S,ref_phase):
    """ 
    This function corrects the absolute phase offset of the polarimetric phase. To do so, it needs a zero-phase reference in the image,
    such as the phase of a trihedral reflector.
    Parameters
    ----------
    S : scatteringMatrix 
        the scattering matrix to be corrected
    ref_phase : double
        Reference phase for the zero offset
    Returns
    -------
    scatteringMatrix
        the scattering matrix with the corrected polarimetric phase
    """
    S_cor = S * 1
    S_cor['VV'] = S['VV'] * np.exp(1j*ref_phase)
    return S_cor

def HH_HV_correlation(S, window = (5,5)):
    """
    This function computes the HH HV correlation for a given scattering matrix
    Parameters
    ----------
    S : scatteringMatrix
    window : tuple
        window size for the correlation computation
    Returns
    -------
    ndarray
        The resulting correlation
    """
    num = corefun.smooth(S['HH'] * S['VV'].conj(), window)
    den = corefun.smooth(np.abs(S['HH'])**2,window) * corefun.smooth(np.abs(S['VV'])**2,window)
    corr = num / np.sqrt(den)
    return corr
    
    
def natural_targets_calibration(S,area,estimation_window):
    """
    This function performs polarimetric calibration using
    the azimuthal symmetry assumption
    Parameters
    ----------
    S : scatteringMatrix
        the image to be calibrated
    area : tuple
        tuple of slice to extract the calibration information
    estimation window : iterable
        tuple or list of window size for the estimation of the correlation
    Returns
    -------
    scatteringMatrix
        the calibrated matrix
    """
    s1 = S[area] * 1
    C = s1.to_coherency_matrix(basis='lexicographic', bistatic=True)
    C = corefun.smooth(C,estimation_window + [1,1])
    delta_0 = C[:,:,0,0]*C[:,:,3,3] - np.abs(C[:,:,0,3])**2
    u_0 = (C[:,:,3,3]*C[:,:,1,0] -C[:,:,3,0]*C[:,:,1,3]) / delta_0
    v_0 = (C[:,:,0,0]*C[:,:,1,3] -C[:,:,1,0]*C[:,:,0,3]) / delta_0
    z_0 = (C[:,:,3,3]*C[:,:,2,0] -C[:,:,3,0]*C[:,:,2,3]) / delta_0
    w_0 = (C[:,:,0,0]*C[:,:,2,3] -C[:,:,2,0]*C[:,:,0,3]) / delta_0
    X_0 = C[:,:,2,1] - z_0 * C[:,:,0,1] - w_0 * C[:,:,3,1]
    alpha_0_1 = (C[:,:,1,1] - u_0 * C[:,:,0,1] - v_0 * C[:,:,3,1]) / X_0
    alpha_0_2 = X_0.conj() / (C[:,:,2,2] - z_0.conj() * C[:,:,2,0] - w_0.conj() * C[:,:,2,3])
    alpha_0 = (np.abs(alpha_0_1*alpha_0_2) - 1 + np.sqrt((np.abs(alpha_0_1*alpha_0_2)-1)**2 + 4 + np.abs(alpha_0_2)**2))/(2*np.abs(alpha_0_2)) * alpha_0_1/np.abs(alpha_0_1)
    alpha = alpha_0
    sigma = np.zeros(alpha.shape + (4,4), dtype = s1.dtype)
    sigma[:,:,0,0] = 1
    sigma[:,:,0,1] = -w_0
    sigma[:,:,0,2] = -v_0
    sigma[:,:,0,3] = v_0*w_0
    sigma[:,:,1,0] = -u_0/np.sqrt(alpha)
    sigma[:,:,1,1] =  1/np.sqrt(alpha)
    sigma[:,:,1,2] =  u_0*v_0/np.sqrt(alpha)
    sigma[:,:,1,3] =  v_0/np.sqrt(alpha)
    sigma[:,:,2,0] = -z_0*np.sqrt(alpha)
    sigma[:,:,2,1] = w_0*z_0*np.sqrt(alpha)
    sigma[:,:,2,2] = np.sqrt(alpha)
    sigma[:,:,2,3] = -w_0*np.sqrt(alpha)
    sigma[:,:,3,0] = u_0*z_0
    sigma[:,:,3,1] = -z_0
    sigma[:,:,3,2] = -u_0
    sigma[:,:,3,3] = 1
    sigma_1 = (np.ones_like(u_0)/((u_0*w_0-1))*(v_0*z_0-1))[:,:,None,None] * sigma
    sigma_1 = np.nanmean(sigma,axis=(0,1))
    sv = S.scattering_vector(basis='lexicographic')
    sv_corr = np.einsum('...ij,...j->...i',sigma_1,sv)
    s_cal = np.reshape(sv_corr,sv_corr.shape[0:2] + (2,2))
    s_cal = s_cal.view(scatteringMatrix)
    s_cal = S.__array_wrap__(s_cal)
    return s_cal, sigma, C

    


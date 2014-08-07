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
import scipy
from scipy import fftpack
#from ..core import scatteringMatrix

def calibrate_from_parameters(S,parameter_name):
    """
    This function performs polarimetric calibration from a text file containing the m-matrix distortion parameters
    Parameters
    ----------
    S : scatteringMatrix
        the matrix to calibrate
    parameter_name : string
        Path to the parameters file, it must in the forma of np.save
    """
    m_inv = np.linalg.pinv(np.load(parameter_name))
    #Imbalance correction
    S_cal = np.einsum('...ij,...j->...i',m_inv,S.scattering_vector(bistatic = True, basis = 'lexicographic'))
    S_cal = S.__array_wrap__(np.reshape(S_cal,S.shape[0:2] + (2,2)))
    return S_cal

def swap_channels(S):
    S1 = S * 1
    S1['HH'] = np.abs(S['VH'])
    S1['VH'] = np.abs(S['HH']) 
    S1['HV'] = np.abs(S['VV'])
    S1['VV'] = np.abs(S['HV'])
    return S1
    

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


def correct_phase_ramp(S,if_phase, conversion_factor = 1, conversion_factor_1 = 1):
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
    S_corr['VV'] = S['VV'] * np.exp(-1j*conversion_factor*if_phase)
    S_corr['HV'] = S['HV'] * np.exp(-1j*conversion_factor_1*if_phase)
    S_corr['VH'] = S['VH'] * np.exp(-1j*conversion_factor_1*if_phase)
    return S_corr
    
def correct_phase_ramp_GPRI(S,S_ref_l,S_ref_u, conversion_factor = 1, conversion_factor_1 = 1):
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
    HH_VV_if = np.angle(S_ref_l['HH']*S_ref_u['HH'].conj())
    S_corr = correct_phase_ramp(S,HH_VV_if, conversion_factor = conversion_factor, conversion_factor_1 = conversion_factor_1)
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



def gct(exact_targets,measured_targets):
    #Matrices
    def sorted_ev(P,N):
        lam_dot,x = np.linalg.eig(P)
        lam, y = np.linalg.eig(N)
        phase_1 = np.abs(np.arctan((lam_dot[0]*lam[1])/(lam_dot[1]*lam[0])))
        phase_2 = np.abs(np.arctan((lam_dot[0]*lam[0])/(lam_dot[1]*lam[1])))
        if phase_2 > phase_1:
            lam = lam[::-1]
            y = y[:,::-1]
        return lam_dot,x,lam,y
            
        
    N1 = measured_targets[0]
    N2 = measured_targets[1]
    N3 = measured_targets[2]
    P1 = exact_targets[0]
    P2 = exact_targets[1]
    P3 = exact_targets[2]
    #similarity transformations
    
    #for transmit distortion
    P_T = np.dot(np.linalg.inv(P1),P2)
    P_T_bar = np.dot(np.linalg.inv(P1),P3)
    
    N_T = np.dot(np.linalg.inv(N1),N2)
    N_T_bar = np.dot(np.linalg.inv(N1),N3)
    
    #for receive distortion
    P_R = np.dot(P2,np.linalg.inv(P1))
    P_R_bar = np.dot(P3,np.linalg.inv(P1))
    
    N_R = np.dot(N2,np.linalg.inv(N1))
    N_R_bar = np.dot(N3,np.linalg.inv(N1))
    
    #eigenvalue decompositions
    
    #for reiceved
    lambda_t_dot,x_t,lambda_t,y_t = sorted_ev(P_T,N_T)
    lambda_t_bar_dot,x_t_bar,lambda_t_bar,y_t_bar = sorted_ev(P_T_bar,N_T_bar)
    #for transmit
    lambda_r_dot,x_r,lambda_r,y_r = sorted_ev(P_R,N_R)
    lambda_r_bar_dot,x_r_bar,lambda_r_bar,y_r_bar = sorted_ev(P_R_bar,N_R_bar)
    
    #Determine T
    #ratio of c1 and c2
    c2_c1 =  ((x_t[0,0]*x_t_bar[1,0] - x_t[1,0]*x_t_bar[0,0]) * (y_t[1,1]*y_t_bar[0,0] - y_t[0,1]*y_t_bar[1,0]))/ \
             ((x_t[1,1]*x_t_bar[0,0] - x_t[0,1]*x_t_bar[1,0]) * (y_t[0,0]*y_t_bar[1,0] - y_t[1,0]*y_t_bar[0,0]))

    #ratio of d1 and d2
    d2_d1 =  ((x_r[0,0]*x_r_bar[1,0] - x_r[1,0]*x_r_bar[0,0]) * (y_r[1,1]*y_r_bar[0,0] - y_r[0,1]*y_r_bar[1,0]))/ \
             ((x_r[1,1]*x_r_bar[0,0] - x_r[0,1]*x_r_bar[1,0]) * (y_r[0,0]*y_r_bar[1,0] - y_r[1,0]*y_r_bar[0,0]))
  

    #C
    c1 = np.linalg.det(y_t) * 1/(x_t[0,0]*y_t[1,1]-c2_c1*x_t[0,1]*y_t[1,0]) 
    c2 = np.linalg.det(y_t) * 1/(1/c2_c1*x_t[0,0]*y_t[1,1]-x_t[0,1]*y_t[1,0])
    #And D
    d1 = np.linalg.det(y_r) * 1/(x_r[0,0]*y_r[1,1]-d2_d1*x_r[0,1]*y_r[1,0]) 
    d2 = np.linalg.det(y_r) * 1/(1/d2_d1*x_r[0,0]*y_r[1,1]-x_r[0,1]*y_r[1,0]) 
    #Determine T and R
    C = np.diag([c1,c2])
    D = np.diag([d1,d2])
    
    T = np.dot(np.dot(x_t,C),np.linalg.inv(y_t))
    R = np.dot(np.dot(x_r,D),np.linalg.inv(y_r))
    return T,R

def distortion_matrices_to_m(R,T):
    a = [[R[0,0]*T[0,0], R[0,0]*T[1,0]+R[0,1]*T[0,0], R[0,1]*T[0,1]],\
        [R[1,0]*T[0,0], R[1,1]*T[0,0], R[1,1]*T[1,0]],\
        [R[0,0]*T[0,1], R[0,0]*T[1,1], R[0,1]*T[1,1]],\
        [R[0,1]*T[0,1], R[1,0]*T[1,1]+R[1,1]*T[0,1], R[1,1]*T[1,1]]]
    M = np.array(a)
    return M
    
def get_shift(image1,image2, oversampling = 1, axes = (0,1)):
    pad_size = zip(np.zeros(image1.ndim),np.zeros(image1.ndim))
    for ax in axes:
        pad_size[ax] = (image1.shape[ax] * oversampling/2,image1.shape[ax] * oversampling/2)
    pad_size = tuple(pad_size)
    image1_pad = np.pad(image1,pad_size,mode='constant')
    image2_pad = np.pad(image2,pad_size,mode='constant')
    corr_image = norm_xcorr(image1_pad,image2_pad, axes = axes)
    shift = np.argmax(np.abs(corr_image))
    shift_idx = np.unravel_index(shift,corr_image.shape)
#    shift_idx = np.array(shift_idx) - corr_image.shape / 2
    shift_idx = tuple(np.divide(np.subtract(shift_idx , np.divide(corr_image.shape,2.0)),oversampling))
    return shift_idx, corr_image

def norm_xcorr(image1,image2, axes = (0,1)):
    image_1_hat = scipy.fftpack.fftn(image1, axes = axes)
    image_2_hat = scipy.fftpack.fftn(image2, axes = axes)
    phase_corr = scipy.fftpack.fftshift(scipy.fftpack.ifftn(image_1_hat * image_2_hat.conj() / (np.abs( image_1_hat * image_2_hat.conj())),axes = axes),axes= axes)
    return phase_corr


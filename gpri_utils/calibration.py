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

def calibrate_from_r_and_t(S, R,T):
    T_inv = np.linalg.inv(T)
    R_inv = np.linalg.inv(R)
    S_cal = corefun.transform(R_inv,S,T_inv)
    return S_cal

def calibrate_from_parameters(S,par):
    """
    This function performs polarimetric calibration from a text file containing the m-matrix distortion parameters
    Parameters
    ----------
    S : scatteringMatrix
        the matrix to calibrate
    par : string, array_like
        Path to the parameters file, it must in the form of np.save
    """
    if  isinstance(par,str):
        m = np.load(parameter_name)
    elif isinstance(par,(np.ndarray,list,tuple)):
        m = par
    else:
        raise TypeError("The parameters must be passed\
        as a list or array or tuple or either as a path to a text file")
    f = m[0]
    g = m[1]
    S_cal = S * 1
    S_cal['VV'] = S_cal['VV'] * 1/f**2
    S_cal['HV'] = S_cal['HV'] * 1/f 
    S_cal['VH'] = S_cal['VH'] * 1/f * 1/g
    return S_cal

def swap_channels(S_u,S_l):
    S_u_n = S_u * 1
    S_l_n = S_l * 1
    S_u_n['VV'] = S_l['VV']
    S_l_n['VV'] = S_u['VV']
#    S_u_n['HH'] = S_l['HH']
#    S_l_n['HH'] = S_u['HH']
    return S_u_n, S_l_n
    

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


def correct_phase_ramp(S,if_phase, cv_vec = [1,0,0]):
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
    S_corr['VV'] = S['VV'] * np.exp(-1j*cv_vec[0]*if_phase)
    S_corr['HV'] = S['HV'] * np.exp(-1j*cv_vec[1]*if_phase)
    S_corr['VH'] = S['VH'] * np.exp(-1j*cv_vec[2]*if_phase)
    return S_corr
    
def correct_phase_ramp_GPRI(S,S_ref_2, conversion_factor = 1, conversion_factor_1 = 0):
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
    S_ref_1 = S * 1
    ref_b = (S_ref_1['HH'].ant_vec - S_ref_2['HH'].ant_vec)
    #Compute conversion factors
    cf_co = (S['HH'].ant_vec - S['VV'].ant_vec) / ref_b
    cf_cr_1 = (S['HH'].ant_vec - S['HV'].ant_vec) / ref_b
    cf_cr_2 = (S['HH'].ant_vec - S['VH'].ant_vec) / ref_b
    HH_VV_if = np.angle(S_ref_1['HH']*S_ref_2['HH'].conj())
    S_corr = correct_phase_ramp(S,HH_VV_if, cv_vec = [cf_co,cf_cr_1,cf_cr_2])
    return S_corr

def correct_phase_ramp_GPRI_DEM(S, DEM, B_if):
    B1 = S.ant_vec[0,0] - S.ant_vec[1,1]
    B2 = S.ant_vec[0,0] - S.ant_vec[0,1]
    B3 = S.ant_vec[0,0] - S.ant_vec[1,0]

    
    ifgram_co = np.exp(-1j * DEM *  B1 / B_if)
    ifgram_HV = np.exp(-1j * DEM * B2 / B_if)
    ifgram_VH = np.exp(-1j * DEM * B3 / B_if)
    
    S_corr = S * 1
    
    S_corr['VV'] = S_corr['VV'] * ifgram_co.conj()
    S_corr['HV'] = S_corr['HV'] * ifgram_HV.conj()
    S_corr['VH'] = S_corr['VH'] * ifgram_VH.conj()
    return S_corr
    
def correct_phase_ramp_DEM(S,DEM):
    B1 = S.ant_vec[0,0] - S.ant_vec[1,1]
    B2 = S.ant_vec[0,0] - S.ant_vec[0,1]
    B3 = S.ant_vec[0,0] - S.ant_vec[1,0]
    if1 = synthetic_interferogram(S,DEM, B1)
    if2 = synthetic_interferogram(S,DEM, B2)
    if3 = synthetic_interferogram(S,DEM, B3)
    S_corr = S * 1
    S_corr['VV'] = S_corr['VV'] * if1.conj()
    S_corr['HV'] = S_corr['HV'] * if2.conj()
    S_corr['VH'] = S_corr['VH'] * if3.conj()
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
    r, th = np.meshgrid(S.r_vec , S.az_vec)
    x =  r * np.cos(th)
    y =  r * np.sin(th)
    z =  DEM
    r1 = np.dstack(( x,  y,  z))
    r2 = np.dstack((x,  y, (z + B)))
    #Convert into 
    delta_d = np.linalg.norm(r1, axis =2) - np.linalg.norm(r2,axis = 2)
    lamb = 3e8/S.center_frequency
    ph = delta_d * 4 / lamb * np.pi 
    return np.exp(1j *ph)

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

def  simple_calibration(S, coord_tri, slice_distributed):
    """
    This function determines the paramrters a
    simple calibration based
    on TCR for imbalance and
    distributed targets for the
    determination of crosspolarized imbalance
    Parameters
    ----------
    S : scatteringMatrix
        The image to calibrate
    coord_tri : tuple
        The coordinates where the TCR is located
    slice_distributed  : tuple
        A tuple of slices identifinyg a region of distributed targets
    """
    #Determine cochannel imbalance
    S_tri = S[coord_tri]
    f_mag = (np.abs(S_tri['VV'])**2 / (np.abs(S_tri['HH'])**2))**0.25
    f_phase = 1 / 2.0 * np.angle(S_tri['HH'].conj() *  S_tri['VV'])
    f = f_mag * np.exp(1j * f_phase)
    S_d = S[slice_distributed]
    g_mag = (np.mean(np.abs(S_d['VH'])**2) / np.mean((np.abs(S_d['HV'])**2)))**0.5
    g_phase =  np.mean(np.angle(S_d['HV'].conj() *  S_d['VH'])) 
    f = f_mag * np.exp(1j * f_phase)
    g = g_mag * np.exp(1j * g_phase)
    #Determine imbalance on natural targets
    return f, g
    

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
            
        
    N1 = np.array(measured_targets[0])
    N2 = np.array(measured_targets[1])
    N3 = np.array(measured_targets[2])
    P1 = np.array(exact_targets[0])
    P2 = np.array(exact_targets[1])
    P3 = np.array(exact_targets[2])
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
    
def get_shift(image1,image2, oversampling = (10,1), axes = (0,1)):
    pad_size = zip(np.zeros(image1.ndim),np.zeros(image1.ndim))
    for ax, ov in zip(axes, oversampling):
        pad_size[ax] = (image1.shape[ax] * (ov - 1),0)
    pad_size = tuple(pad_size)
    image1_pad = np.pad(image1,pad_size,mode='constant')
    image2_pad = np.pad(image2,pad_size,mode='constant')
#    x = np.arange(image1.shape[0] * oversampling[0]) / oversampling[0]
#    y = np.arange(image1.shape[1] * oversampling[1]) / oversampling[1]
#    x, y = np.meshgrid(x,y)
#    import pyrat.visualization.visfun as visfun
#    image1_pad = visfun.bilinear_interpolate(image1, x.T, y.T)
#    image2_pad = visfun.bilinear_interpolate(image2, x.T, y.T)
#    image1_pad = scipy.ndimage.zoom(image1, oversampling, order=1)
#    image2_pad = scipy.ndimage.zoom(image2, oversampling, order=1)
    corr_image = norm_xcorr(image1_pad, image2_pad, axes = axes)
    shift = np.argmax(np.abs(corr_image))
    shift_idx = np.unravel_index(shift,corr_image.shape)
    shift_idx = (np.subtract(np.array(shift_idx) , np.divide(corr_image.shape , 2.0)))
    return shift_idx, corr_image

def norm_xcorr(image1,image2, axes = (0,1)):
    import pyfftw
    image1[np.isnan(image1)] = 0
    image2[np.isnan(image2)] = 0
    image_1_hat = pyfftw.interfaces.scipy_fftpack.fftn(image1, axes = axes)
    image_2_hat = pyfftw.interfaces.scipy_fftpack.fftn(image2, axes = axes)
    phase_corr = scipy.fftpack.fftshift(pyfftw.interfaces.scipy_fftpack.ifftn(image_1_hat * image_2_hat.conj() / (np.abs( image_1_hat * image_2_hat.conj())),axes = axes),axes= axes)
    return phase_corr


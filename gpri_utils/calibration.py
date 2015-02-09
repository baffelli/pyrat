# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:56:18 2014

@author: baffelli
"""

"""
Utilities for GPRI calibration
"""
import numpy as _np
from pyrat import core, matrices
from ..core import corefun, polfun
import scipy as _sc
from scipy import fftpack as _fftp
from scipy import signal as _sg
#from ..core import scatteringMatrix

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


def coherency_matrix_to_scattering_matrix(C):
    if C.basis == 'lexicographic':
        pass
    else:
        C = C.pauli_to_lexicographic()
    if C.ndim == 2:
        S = _np.zeros((2,2), dtype = C.dtype)
        sl = ()
    elif C.ndim == 4:
        S = _np.zeros(C.shape[0:2] + (2,2), dtype = C.dtype)
        sl = (Ellipsis, Ellipsis)
    S = matrices.scatteringMatrix(S)
    S['HH'] = _np.sqrt(C[sl + (0,0)])
    S['HV'] = _np.sqrt(C[sl + (1,1)]) * _np.exp(1j * _np.angle(C[sl + (1,0)]))
    S['VH'] = _np.sqrt(C[sl + (2,2)]) * _np.exp(1j * _np.angle(C[sl + (2,0)]))
    S['VV'] = _np.sqrt(C[sl + (3,3)]) * _np.exp(1j * _np.angle(C[sl + (3,0)]))
    S.__dict__ = C.__dict__
    S.basis = []
    return S

def remove_phase_ramp(S1, B_if, ph_if, bistatic = False, S2 = None):
    if S2 is None:
        S2 = S1 * 1
    else:
        pass
    k1 = S1.scattering_vector(basis = 'lexicographic', bistatic = bistatic)
    k2 = S2.scattering_vector(basis = 'lexicographic', bistatic = bistatic)
    C = corefun.outer_product(k1, k2)
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
    C_cal = C.__array_wrap__(C_cal)
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
    co_shift, corr_co = corefun.get_shift(_np.abs(S['HH'][shift_patch]),_np.abs(S['VV'][shift_patch]), axes = (0,1), oversampling = 1)
    cross_shift, cross_co = corefun.get_shift(_np.abs(S['HH'][shift_patch]),_np.abs(S['HV'][shift_patch]), axes = (0,1), oversampling = 1)
    cross_shift_1, cross_co = corefun.get_shift(_np.abs(S['HH'][shift_patch]),_np.abs(S['VH'][shift_patch]), axes = (0,1), oversampling = 1)
    S_cor['VV'] = corefun.shift_array(S['VV'],(co_shift[0],0))
    S_cor['HV'] = corefun.shift_array(S['HV'],(cross_shift[0],0))
    S_cor['VH'] = corefun.shift_array(S['VH'],(cross_shift_1[0],0))
    return S_cor


def correct_HHVV_phase_ramp(S, if_phase, baseline_ratio):
    """
    This function corrects the phase ramp
    between the HH-VV channels. It is used
    as a preliminary correction, useful
    to determine the correct HH-VV
    phase
    Parameters
    ----------
    S : scatteringMatrix
        `S` is the Scattering matrix that is to be corrected
    if_phase : array_like
        `if_phase` the unwrapped interferometric phase
        which is to be removed from S
    baseline_ratio  : float
        The ratio of the polarimetric baseline to
        the interferometric baseline
    Returns
    -------
    S_coor : scatteringMatrix
        A scattering matrix where the HH-VV phase contains a
        polarimetric contribution only
    """
    S_corr = S * 1
    
    S_corr['VV'] = S['VV'] *  _np.exp(-1j*baseline_ratio*if_phase)
    return S_corr
        

        
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
    den = corefun.smooth(_np.abs(S['HH'])**2,window) * corefun.smooth(_np.abs(S['VV'])**2,window)
    corr = num / _np.sqrt(den)
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
    delta_0 = C[:,:,0,0]*C[:,:,3,3] - _np.abs(C[:,:,0,3])**2
    u_0 = (C[:,:,3,3]*C[:,:,1,0] -C[:,:,3,0]*C[:,:,1,3]) / delta_0
    v_0 = (C[:,:,0,0]*C[:,:,1,3] -C[:,:,1,0]*C[:,:,0,3]) / delta_0
    z_0 = (C[:,:,3,3]*C[:,:,2,0] -C[:,:,3,0]*C[:,:,2,3]) / delta_0
    w_0 = (C[:,:,0,0]*C[:,:,2,3] -C[:,:,2,0]*C[:,:,0,3]) / delta_0
    X_0 = C[:,:,2,1] - z_0 * C[:,:,0,1] - w_0 * C[:,:,3,1]
    alpha_0_1 = (C[:,:,1,1] - u_0 * C[:,:,0,1] - v_0 * C[:,:,3,1]) / X_0
    alpha_0_2 = X_0.conj() / (C[:,:,2,2] - z_0.conj() * C[:,:,2,0] - w_0.conj() * C[:,:,2,3])
    alpha_0 = (_np.abs(alpha_0_1*alpha_0_2) - 1 + _np.sqrt((_np.abs(alpha_0_1*alpha_0_2)-1)**2 + 4 + _np.abs(alpha_0_2)**2))/(2*_np.abs(alpha_0_2)) * alpha_0_1/_np.abs(alpha_0_1)
    alpha = alpha_0
    sigma = _np.zeros(alpha.shape + (4,4), dtype = s1.dtype)
    sigma[:,:,0,0] = 1
    sigma[:,:,0,1] = -w_0
    sigma[:,:,0,2] = -v_0
    sigma[:,:,0,3] = v_0*w_0
    sigma[:,:,1,0] = -u_0/_np.sqrt(alpha)
    sigma[:,:,1,1] =  1/_np.sqrt(alpha)
    sigma[:,:,1,2] =  u_0*v_0/_np.sqrt(alpha)
    sigma[:,:,1,3] =  v_0/_np.sqrt(alpha)
    sigma[:,:,2,0] = -z_0*_np.sqrt(alpha)
    sigma[:,:,2,1] = w_0*z_0*_np.sqrt(alpha)
    sigma[:,:,2,2] = _np.sqrt(alpha)
    sigma[:,:,2,3] = -w_0*_np.sqrt(alpha)
    sigma[:,:,3,0] = u_0*z_0
    sigma[:,:,3,1] = -z_0
    sigma[:,:,3,2] = -u_0
    sigma[:,:,3,3] = 1
    sigma_1 = (_np.ones_like(u_0)/((u_0*w_0-1))*(v_0*z_0-1))[:,:,None,None] * sigma
    sigma_1 = _np.nanmean(sigma,axis=(0,1))
    sv = S.scattering_vector(basis='lexicographic')
    sv_corr = _np.einsum('...ij,...j->...i',sigma_1,sv)
    s_cal = _np.reshape(sv_corr,sv_corr.shape[0:2] + (2,2))
    s_cal = s_cal.view(scatteringMatrix)
    s_cal = S.__array_wrap__(s_cal)
    return s_cal, sigma, C

def  simple_calibration(C_tri,C_distributed):
    """
    This function determines the paramrters a
    simple calibration based
    on TCR for imbalance and
    distributed targets for the
    determination of crosspolarized imbalance
    Parameters
    ----------
    C : coherencyMatrix
        The image to calibrate
    coord_tri : tuple
        The coordinates where the TCR is located
    slice_distributed  : tuple
        A tuple of slices identifinyg a region of distributed targets
    """
    if isinstance(C_tri, core.matrices.coherencyMatrix) and C_tri.basis == 'lexicographic':
        #Determine cochannel imbalance
        f_mag = (_np.abs(C_tri[3,3]) / (_np.abs(C_tri[0,0])))**0.25
        f_phase = 1/2.0* _np.angle(C_tri[3,0])
        f = f_mag * _np.exp(1j * f_phase)
        #Determine for distributed targets
        C_d = C_distributed
        g_mag = _np.mean(_np.abs(C_d[2,2])) / _np.mean((_np.abs(C_d[1,1])))
#        g_mag = (_np.mean(_np.abs(S_d['VH'])**2) / _np.mean((_np.abs(S_d['HV'])**2)))**0.5
#        g_phase =  _np.mean(_np.angle(S_d['HV'].conj() *  S_d['VH'])) 
        g_phase =  _np.mean(_np.angle(C_d[2,1]))
        f = f_mag * _np.exp(1j * f_phase)
        g = g_mag * _np.exp(1j * g_phase)
        #Determine imbalance on natural targets
        return f, g
    

def gct(exact_targets,measured_targets):
    #Matrices
    def sorted_ev(P,N):
        lam_dot,x = _np.linalg.eig(P)
        lam, y = _np.linalg.eig(N)
        phase_1 = _np.abs(_np.arctan((lam_dot[0]*lam[1])/(lam_dot[1]*lam[0])))
        phase_2 = _np.abs(_np.arctan((lam_dot[0]*lam[0])/(lam_dot[1]*lam[1])))
        if phase_2 > phase_1:
            lam = lam[::-1]
            y = y[:,::-1]
        return lam_dot,x,lam,y
            
        
    N1 = _np.array(measured_targets[0])
    N2 = _np.array(measured_targets[1])
    N3 = _np.array(measured_targets[2])
    P1 = _np.array(exact_targets[0])
    P2 = _np.array(exact_targets[1])
    P3 = _np.array(exact_targets[2])
    #similarity transformations
    
    #for transmit distortion
    P_T = _np.dot(_np.linalg.inv(P1),P2)
    P_T_bar = _np.dot(_np.linalg.inv(P1),P3)
    
    N_T = _np.dot(_np.linalg.inv(N1),N2)
    N_T_bar = _np.dot(_np.linalg.inv(N1),N3)
    
    #for receive distortion
    P_R = _np.dot(P2,_np.linalg.inv(P1))
    P_R_bar = _np.dot(P3,_np.linalg.inv(P1))
    
    N_R = _np.dot(N2,_np.linalg.inv(N1))
    N_R_bar = _np.dot(N3,_np.linalg.inv(N1))
    
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
    c1 = _np.linalg.det(y_t) * 1/(x_t[0,0]*y_t[1,1]-c2_c1*x_t[0,1]*y_t[1,0]) 
    c2 = _np.linalg.det(y_t) * 1/(1/c2_c1*x_t[0,0]*y_t[1,1]-x_t[0,1]*y_t[1,0])
    #And D
    d1 = _np.linalg.det(y_r) * 1/(x_r[0,0]*y_r[1,1]-d2_d1*x_r[0,1]*y_r[1,0]) 
    d2 = _np.linalg.det(y_r) * 1/(1/d2_d1*x_r[0,0]*y_r[1,1]-x_r[0,1]*y_r[1,0]) 
    #Determine T and R
    C = _np.diag([c1,c2])
    D = _np.diag([d1,d2])
    
    T = _np.dot(_np.dot(x_t,C),_np.linalg.inv(y_t))
    R = _np.dot(_np.dot(x_r,D),_np.linalg.inv(y_r))
    return T,R



    
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
#    shift = _np.mod(.astype(_np.int))
#    shift_idx = shift / _np.array(oversampling)
#    shift_idx = _np.array(shift_idx) / _np.array(oversampling).astype(_np.float)
    return shift_idx, corr_image
    
    
def norm_xcorr(image1,image2, axes = (0,1),  oversampling = (2,2)):
    import pyfftw.interfaces.scipy_fftpack as fftp
    image1[_np.isnan(image1)] = 0
    image2[_np.isnan(image2)] = 0
    #Take trnasform
    image_1_hat = fftp.fftn(image1, axes = axes)
    image_2_hat = fftp.fftn(image2, axes = axes)
    #Oversample
    pad_size = zip(_np.zeros(image1.ndim),_np.zeros(image1.ndim))
    for ax, ov in zip(axes, oversampling):
        pad_size[ax] = (image1.shape[ax] * (ov - 1), 0)
    pad_size = tuple(pad_size)
    ft_corr = image_1_hat * image_2_hat.conj()\
    / (_np.abs( image_1_hat * image_2_hat.conj()))
    ft_corr_pad = _np.pad(ft_corr, pad_size, mode='constant')
    phase_corr = fftp.fftshift(fftp.ifftn(ft_corr_pad,axes = axes)\
    ,axes = axes)
    return phase_corr


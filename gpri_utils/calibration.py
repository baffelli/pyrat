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
from ..visualization import visfun as _vf
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
    print co_shift, cross_shift, cross_shift_1
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
    # if isinstance(C_tri, core.matrices.coherencyMatrix) and C_tri.basis == 'lexicographic':
    #Determine cochannel imbalance
    f_mag = (_np.abs(C_tri[3,3]) / (_np.abs(C_tri[0,0])))**0.25
    f_phase = 1/2.0 * _np.angle(C_tri[3,0])
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
    

#def gct(exact_targets,measured_targets):
#    #Matrices
#    def sorted_ev(P,N):
#        lam_dot,x = _np.linalg.eig(P)
#        lam, y = _np.linalg.eig(N)
#        phase_1 = _np.abs(_np.arctan((lam_dot[0]*lam[1])/(lam_dot[1]*lam[0])))
#        phase_2 = _np.abs(_np.arctan((lam_dot[0]*lam[0])/(lam_dot[1]*lam[1])))
#        if phase_2 > phase_1:
#            lam = lam[::-1]
#            y = y[:,::-1]
#        return lam_dot,x,lam,y
#            
#        
#    N1 = _np.array(measured_targets[0])
#    N2 = _np.array(measured_targets[1])
#    N3 = _np.array(measured_targets[2])
#    P1 = _np.array(exact_targets[0])
#    P2 = _np.array(exact_targets[1])
#    P3 = _np.array(exact_targets[2])
#    #similarity transformations
#    
#    #for transmit distortion
#    P_T = _np.dot(_np.linalg.inv(P1),P2)
#    P_T_bar = _np.dot(_np.linalg.inv(P1),P3)
#    
#    N_T = _np.dot(_np.linalg.inv(N1),N2)
#    N_T_bar = _np.dot(_np.linalg.inv(N1),N3)
#    
#    #for receive distortion
#    P_R = _np.dot(P2,_np.linalg.inv(P1))
#    P_R_bar = _np.dot(P3,_np.linalg.inv(P1))
#    
#    N_R = _np.dot(N2,_np.linalg.inv(N1))
#    N_R_bar = _np.dot(N3,_np.linalg.inv(N1))
#    
#    #eigenvalue decompositions
#    
#    #for reiceved
#    lambda_t_dot,x_t,lambda_t,y_t = sorted_ev(P_T,N_T)
#    lambda_t_bar_dot,x_t_bar,lambda_t_bar,y_t_bar = sorted_ev(P_T_bar,N_T_bar)
#    #for transmit
#    lambda_r_dot,x_r,lambda_r,y_r = sorted_ev(P_R,N_R)
#    lambda_r_bar_dot,x_r_bar,lambda_r_bar,y_r_bar = sorted_ev(P_R_bar,N_R_bar)
#    
#    #Determine T
#    #ratio of c1 and c2
#    c2_c1 =  ((x_t[0,0]*x_t_bar[1,0] - x_t[1,0]*x_t_bar[0,0]) * (y_t[1,1]*y_t_bar[0,0] - y_t[0,1]*y_t_bar[1,0]))/ \
#             ((x_t[1,1]*x_t_bar[0,0] - x_t[0,1]*x_t_bar[1,0]) * (y_t[0,0]*y_t_bar[1,0] - y_t[1,0]*y_t_bar[0,0]))
#
#    #ratio of d1 and d2
#    d2_d1 =  ((x_r[0,0]*x_r_bar[1,0] - x_r[1,0]*x_r_bar[0,0]) * (y_r[1,1]*y_r_bar[0,0] - y_r[0,1]*y_r_bar[1,0]))/ \
#             ((x_r[1,1]*x_r_bar[0,0] - x_r[0,1]*x_r_bar[1,0]) * (y_r[0,0]*y_r_bar[1,0] - y_r[1,0]*y_r_bar[0,0]))
#  
#
#    #C
#    c1 = _np.linalg.det(y_t) * 1/(x_t[0,0]*y_t[1,1]-c2_c1*x_t[0,1]*y_t[1,0]) 
#    c2 = _np.linalg.det(y_t) * 1/(1/c2_c1*x_t[0,0]*y_t[1,1]-x_t[0,1]*y_t[1,0])
#    #And D
#    d1 = _np.linalg.det(y_r) * 1/(x_r[0,0]*y_r[1,1]-d2_d1*x_r[0,1]*y_r[1,0]) 
#    d2 = _np.linalg.det(y_r) * 1/(1/d2_d1*x_r[0,0]*y_r[1,1]-x_r[0,1]*y_r[1,0]) 
#    #Determine T and R
#    C = _np.diag([c1,c2])
#    D = _np.diag([d1,d2])
#    
#    T = _np.dot(_np.dot(x_t,C),_np.linalg.inv(y_t))
#    R = _np.dot(_np.dot(x_r,D),_np.linalg.inv(y_r))
#    return T,R



    
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



def rep_2(r_arm, r_ph, r_sl, theta, wrap = True):
    """
    This function computes the phase caused by a shifted
    phase center in the antenna
    """
    lam = (3e8) /17.2e9
    ant_angle = _np.arctan2(r_ph, r_arm)
    r_ant = _np.sqrt(r_arm**2 + r_ph**2)
    # #angle beta
    #Chord length
    c = 2 * r_ant * _np.sin(theta/2)
    mixed_term = 2 * c * r_sl * _np.cos(_np.pi/2 - ant_angle - theta/2)
    dist = _np.sqrt(c**2 + r_sl**2 - mixed_term)

    if wrap is True :
        return _np.mod(-4 * _np.pi * dist/lam, 2 * _np.pi), dist
    else:
        return (-4 * _np.pi * dist/lam), dist

def distance_from_phase_center(r_arm, r_ph, r_sl, theta, wrap=True):
    """
    This function computes the phase caused by a shifted
    phase center in the antenna
    """
    lam = (299792458) /17.2e9
    ant_angle = _np.arctan2(r_ph, r_arm)
    r_ant = _np.sqrt(r_arm**2 + r_ph**2)
    #Chord length
    c = r_ant + r_sl
    mixed_term = 2 * c * r_ant * _np.cos(theta - ant_angle)
    dist = _np.sqrt(c**2 + r_ant**2 - mixed_term)
    if wrap is True :
        return _np.mod(-4 * _np.pi * dist/lam, 2 * _np.pi), dist
    else:
        return (-4 * _np.pi * dist/lam), dist

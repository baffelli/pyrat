import sys
import os
sys.path.append(os.path.expanduser('~/PhD/trunk/Code/'))
import numpy as np
import pyrat as pt
import pyrat.gpri_utils.calibration as cal
import pyrat.visualization.visfun as vf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy as sp
from scipy import signal as sig
import scipy.optimize as opt


C = 299792458.0    #speed of light m/s
KU_WIDTH = 15.798e-3 #WG-62 Ku-Band waveguide width dimension
KU_DZ = 10.682e-3   #Ku-Band Waveguide slot spacing
RANGE_OFFSET= 3

rawdata = pt.gpri_files.rawData(raw_par_path, raw_path)



#Constructs a slice around a point
def slice_around(coord, len):
    return slice(coord - int(len/2), coord + int(len/2))

def lamg(freq, w):
    """
    This function computes the wavelength in waveguide for the TE10 mode
    """
    la = lam(freq)
    return la / np.sqrt(1.0 - (la / (2 * w))**2)	#wavelength in WG-62 waveguide

#lambda in freespace
def lam(freq):
    """
    This function computes the wavelength in freespace
    """
    return C/freq

def squint_angle(freq, w, s):
    """
    This function computes the direction of the main lobe of a slotted
    waveguide antenna as a function of the frequency, the size and the slot spacing.
    It supposes a waveguide for the TE10 mode
    """
    sq_ang = np.arccos(lam(freq) / lamg(freq, w) -  lam(freq) / (2 * s))
    dphi = np.pi *(2.*s/lamg(freq, w) - 1.0)				#antenna phase taper to generate squint
    sq_ang_1 = np.rad2deg(np.arcsin(lam(freq) *dphi /(2.*np.pi*s)))	#azimuth beam squint angle
    return sq_ang_1


def correct_squint(arr, squint_vec, angle_vec):
    """
    This function corrects the frequency dependent squint
    in the GPRI data
    """
    arr = arr[1:,:]
    assert len(squint_vec) == arr.shape[0]
    arr_int = np.zeros_like(arr)
    for idx in range(arr.shape[0]):
        if idx % 100 == 0:
            print("interp range:" + str(idx))
        az_new = angle_vec + squint_vec[idx]
        arr_int[idx,: ] = np.interp(az_new, angle_vec, arr[idx,:],left=0.0, right=0.0)
    return arr_int


def weight_zero(arr,n_zero ):
    """
    This function weights the response close to the zeroth samples, where the sweep stops abrutply
    :return:
    """
    arr_zero = np.array(arr, copy=True)
    win = sp.hanning(2*n_zero)
    for idx_az in range(arr.shape[1]):
        arr_zero[0:n_zero, idx_az] = arr[0:n_zero, idx_az] * win[0:n_zero]
        arr_zero[-n_zero:, idx_az] = arr[-n_zero:, idx_az] * win[n_zero:]
    return arr_zero

def decimate(arr,dec):
    dec_arr = np.zeros([arr.shape[0], arr.shape[1]/dec])
    for i in range(0, dec_arr.shape[1]):
        for j in range(dec):
            dec_arr[:,i] =+ arr[:, i * dec + j]
    dec_arr = dec_arr[:,:] / dec
    return dec_arr


def compress(arr, par, filt_win, apply_scale=True):
    arr_compr = np.zeros((par['CHP_num_samp']/2 + 1, arr.shape[1]) ,dtype=np.complex64)
    win = sp.kaiser(par['CHP_num_samp'], filt_win)
    fshift = np.ones((par['CHP_num_samp']/ 2 +1))
    fshift[1::2] = -1
    #Range video phase
    # rvp = np.exp(1j*4.*np.pi*par['RF_chirp_rate']*(slr/C)**2) #range video phase correction
    nsamp = par['CHP_num_samp']
    pn1 = np.arange(nsamp/2 + 1)		#list of slant range pixel numbers
    rps = (par['ADC_sample_rate']/nsamp*C/2.)/par['RF_chirp_rate'] #range pixel spacing
    slr = (pn1 * par['ADC_sample_rate']/nsamp*C/2.)/par['RF_chirp_rate']  + RANGE_OFFSET  #slant range for each sample
    scale = (np.abs(slr)/slr[nsamp/8])**1.5     #cubic range weighting in power
    for idx_az in range(arr.shape[1]):
        arr_compr[:, idx_az] = np.fft.rfft(arr[:, idx_az] * win, axis=0) * fshift
        if apply_scale:
            arr_compr[:, idx_az] = arr_compr[:, idx_az] * scale
    #Compute range and azimuth vectors
    tcycle = arr.shape[0] * 1/par['ADC_sample_rate']
    azspacing = tcycle * par['STP_rotation_speed'] * 4
    az_vec = np.arange(arr_compr.shape[1]) * azspacing + par['STP_antenna_start']
    return arr_compr, slr, az_vec






HH = rawdata.get_channel('AAA', 'u')
VV = rawdata.get_channel('BBB', 'u')



VV_corr_path = '/data/HIL/20140910/corr/HH_corr'
HH_corr_path = '/data/HIL/20140910/corr/VV_corr'

try:
    dt = pt.gpri_files.type_mapping['SHORT INTEGER']
    corr_shape = (VV.shape[0] - 1, VV.shape[1])
    VV_corr = np.fromfile(VV_corr_path,dtype=dt).reshape(corr_shape)
    HH_corr = np.fromfile(HH_corr_path,dtype=dt).reshape(corr_shape)
except:
    #
    ang_vec = np.arange(HH.shape[1])
    sq_ang = squint_angle(HH.freq_vec, KU_WIDTH, KU_DZ)
    sq_vec = (sq_ang - sq_ang[sq_ang.shape[0]/2]) / HH.az_spacing

    VV_corr = correct_squint(VV, sq_vec - 0.268 / HH.az_spacing  , ang_vec)
    HH_corr = correct_squint(HH, sq_vec, ang_vec)
    HH_corr.T.astype(pt.gpri_files.type_mapping['SHORT INTEGER']).tofile('/data/HIL/20140910/raw/HH_corr')
    VV_corr.T.astype(pt.gpri_files.type_mapping['SHORT INTEGER']).tofile('/data/HIL/20140910/raw/VV_corr')





#Estimate shift


#Reference range
refr = 537
VV_compr, r_vec, az_vec = compress(VV_corr, VV.__dict__,1, 3)
#Find the maximum
max_idx = np.argmax(VV_compr[refr,:])
#Determine 3db region
half_pwr_idx = np.nonzero(np.abs(VV_compr[refr,:]) > np.abs(VV_compr[refr,max_idx]) * 0.9)
#Construct a slice object for plotting etc
sl = [refr, half_pwr_idx]
#Plot
plt.subplot(2,1,1)
plt.plot((np.abs(VV_compr[sl].squeeze())))
plt.subplot(2,1,2)
plt.plot((np.angle(VV_compr[sl].squeeze())))
#Rotation arm of antenna
r_arm = 0.25
#Determine parameters
def cf(r_arm, r_ph, r, az_vec, off, meas_phase):
    sim_phase, dist = cal.rep_2(r_arm, r_ph, r, az_vec, wrap = False)
    cost = np.mean(np.abs(meas_phase - (sim_phase + off))**2)
    return cost

az_VV = np.deg2rad(az_vec[sl[1]] - az_vec[max_idx])
ph_VV = np.unwrap(np.angle(VV_compr[sl].squeeze()))
cost_VV = lambda par_vec: cf(r_arm, par_vec[0], r_vec[refr],az_VV, par_vec[1], ph_VV)
res = opt.minimize(cost_VV, [0,0])

ph_VV_sim, dist_VV_sim = cal.rep_2(r_arm, res['x'][0], r_vec[refr], az_VV, wrap=False)


#Now, we can determine a filter

filt = np.zeros(VV_compr.shape[1], dtype=np.complex64)
filt_kern = ph_VV_sim[0:20]
start_filt_idx = (VV_compr.shape[1]/2 - filt_kern.shape[0]/2)
end_filt_idx = start_filt_idx + len(filt_kern)
filt[start_filt_idx:end_filt_idx] = np.exp(1j * (filt_kern + res['x'][1])) * np.hamming(filt_kern.shape[0])

VV_corr = np.convolve(VV_compr[refr,:], filt, mode='same')

plt.subplot(2,1,1)
plt.plot((np.abs(VV_compr[sl].squeeze())))
plt.plot((np.abs(VV_corr[sl[1]])))
plt.subplot(2,1,2)
plt.plot((np.angle(VV_compr[sl].squeeze())))
plt.plot((np.angle(VV_corr[sl[1]])))


VV_corr = VV_compr[0:1000,:] * 1
for idx_r in range(VV_corr.shape[0]):
    VV_corr[idx_r,:] = sp.signal.fftconvolve(VV_compr[idx_r,:], filt, mode='same')
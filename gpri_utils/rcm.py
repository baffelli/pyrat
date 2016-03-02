import pyrat.fileutils.gpri_files as gpf
import numpy as np
import matplotlib.pyplot as plt
import pyrat.visualization.visfun as vf
import scipy.ndimage as nd
import pyrat.gpri_utils.calibration as cal
import scipy.signal as sig

#Chirp parameters
C = 299792458
f0 = 1.72e10
bw = 200e6
chirp_length = 1e-3
chirp_rate = bw / chirp_length
adc_rate = 6.25000e+06
#Target parameters
r_c = 550
r_arm = np.sqrt(0.25**2 + 0.15**2)
r_arm = 20
n_lines = 1500
omega = np.deg2rad(1.2)
ant_bw = np.deg2rad(0.04)
tau_ant = ant_bw / omega
#Number of samples to zero out
z = 200


#Simulate the fmcw signal for a given range
def fmcw_signal(r, t_chirp, chirp_rate, adc_rate, f0):
    #Number of samples in chirp
    n_chirp = adc_rate * t_chirp
    #Fast time vector
    fast_time = np.arange(n_chirp) * 1/adc_rate
    #Dechirped signal
    lam = C / f0
    #Propagation phase
    signal = np.exp(-1j * 4 * np.pi * r * f0 / C) * \
             np.exp(-1j * np.pi * 4 * r/C * chirp_rate * fast_time) * \
              np.exp(-1j * 4 * np.pi * (r/C)**2 * chirp_rate)
    return signal

#Simulate distance from a target
def distance_from_target(r_c, r_ant, tau, tau_target, omega):
    return np.sqrt(r_c**2 + r_ant**2 - 2 * r_c * r_ant * np.cos(omega * (tau - tau_target)))


#Combine into scanning signal
def scanning_signal(t_chirp, chirp_rate, adc_rate, f0, r_c, r_ant, tau_target, omega, nlines):
    #Number of samples in chirp
    n_chirp = adc_rate * t_chirp
    print(n_chirp)
    #Fast time vector
    fast_time = np.arange(n_chirp) * 1/adc_rate
    #Slow time vector
    slow_time = np.arange(0,n_lines) * t_chirp
    #Dechirped signal
    lam = C / f0
    #Signal vector
    sig_vector = np.zeros((n_chirp, nlines), dtype=np.complex64)
    chirp_freq = f0 - 100e6 + chirp_rate * fast_time
    #Squint vector
    squint_ang = np.deg2rad(gpf.squint_angle(chirp_freq, gpf.KU_WIDTH, gpf.KU_DZ))
    squint_ang = squint_ang - squint_ang[squint_ang.shape[0]/2]
    squint_t = squint_ang / omega
    plt.show()
    #Propagation phase
    for idx_t, t in enumerate(slow_time):
        tau = fast_time + t
        r = distance_from_target(r_c, r_ant, tau + squint_t, tau_target, omega)
        signal = np.exp(-1j * 4 * np.pi * r * f0 / C) * \
             np.exp(-1j * np.pi * 4 * r/C * chirp_rate * fast_time) *\
            np.exp(1j * np.pi * 2*  (r/C)**2 * chirp_rate)
        #Antenna pattern
        ant_pat = np.sinc(((tau - tau_target + squint_t)) / tau_ant)**3 * 10
        sig_vector[:, idx_t] = signal * ant_pat
    return np.real(sig_vector)


def correct_squint(signal, t_chirp, chirp_rate, adc_rate, f0, omega):
    signal_corr = signal * 1
    nlines = signal.shape[1]
    n_chirp = adc_rate * t_chirp
    #Fast time vector
    fast_time = np.arange(n_chirp) * 1/adc_rate
    #Slow time vector
    slow_time = np.arange(0,n_lines) * t_chirp
    #Dechirped signal
    lam = C / f0
    #Signal vector
    sig_vector = np.zeros((n_chirp, nlines), dtype=np.complex64)
    chirp_freq = f0 - bw/2 + chirp_rate * fast_time
    #Squint vector
    squint_ang = np.deg2rad(gpf.squint_angle(chirp_freq, gpf.KU_WIDTH, gpf.KU_DZ))
    squint_ang = squint_ang - squint_ang[squint_ang.shape[0]/2]
    squint_t = squint_ang / omega
    for idx_freq in range(signal.shape[0]):
        az_vec = np.arange(signal.shape[1])
        az_vec_corr = az_vec + squint_t[idx_freq] / t_chirp
        print(az_vec_corr)
        signal_corr[idx_freq,:] = np.interp(az_vec, az_vec_corr, signal[idx_freq,:].real)
    return signal_corr


s = scanning_signal(chirp_length, chirp_rate, adc_rate, f0, r_c, r_arm, chirp_length * 500, omega,n_lines)
#Correct antenna squint
s_desq = correct_squint(s, chirp_length, chirp_rate, adc_rate, f0, omega)

r_win = sig.kaiser(s.shape[0], 23)
zero = np.ones(s.shape[0])
zero[0:z] = np.hamming(2 * z)[0:z]
zero[-z:] = np.hamming(2 * z)[-z:]
fshift = np.ones(s.shape[0]/2 +1)
fshift[0::2] = -1
s_compr = np.fft.rfft(s * r_win[:,None] * zero[:,None], axis =0 ) * fshift[:, None]
s_compr_desq = np.fft.rfft(s_desq * r_win[:,None] * zero[:,None], axis =0 ) * fshift[:, None]
rgb, pal, crap = vf.dismph(s_compr, k=0.3)
rgb_corr, pal, crap = vf.dismph(s_compr_desq, k=0.3)

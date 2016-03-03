import pyrat.fileutils.gpri_files as gpf
import numpy as np
import matplotlib.pyplot as plt
import pyrat.visualization.visfun as vf
import scipy.ndimage as nd
import pyrat.gpri_utils.calibration as cal
import scipy.signal as sig



def distance_from_target(r_c, r_ant, tau, tau_target, omega):
    return np.sqrt(r_c**2 + r_ant**2 - 2 * r_c * r_ant * np.cos(omega * (tau - tau_target)))

class acquisitionSimulator:
    def __init__(self, prf_file, ant_file, reflector_list, raw_file, raw_par_file):
        acquisition_params = gpf.par_to_dict(prf_file)
        antenna_params = gpf.par_to_dict(ant_file)
        #Parameters relating to chirp and scanning
        self.number_of_chirp_samples = acquisition_params['CHP_num_samp']
        self.chirp_duration = acquisition_params['CHP_num_samp'] / acquisition_params['ADC_sample_rate']
        self.f0 = acquisition_params['RF_center_freq']
        self.bw = acquisition_params['CHP_freq_max'] - acquisition_params['CHP_freq_min']
        self.k = self.bw / self.chirp_duration#chirp rate in hz/sec
        self.az_min =  np.deg2rad(acquisition_params['STP_antenna_start'])
        self.az_max = np.deg2rad(acquisition_params['STP_antenna_end'])
        self.omega = np.deg2rad(acquisition_params['STP_rotation_speed'])
        self.ADC_capture_time = np.abs(self.az_min - self.az_max) / self.omega#total slow time
        #Add the capture time due to the ramp
        (ang, rate_max, t_acc, ang_acc, rate_max) = gpf.ant_pos(0.0, acquisition_params['STP_antenna_start'], acquisition_params['STP_antenna_end'],
                                                              acquisition_params['STP_gear_ratio'],acquisition_params['STP_rotation_speed'])
        self.ADC_capture_time = self.ADC_capture_time + 2 * t_acc
        self.nl_tot = self.ADC_capture_time / self.chirp_duration#total number of azimuth lines
        self.ADC_sample_rate = acquisition_params['ADC_sample_rate']
        self.chirp_freq = self.f0  + np.linspace(acquisition_params['CHP_freq_min'], acquisition_params['CHP_freq_max'], acquisition_params['CHP_num_samp'])
        #Antenna parameters
        self.squint_rate = np.deg2rad(antenna_params['antenna_squint_rate'])
        self.r_arm = np.sqrt(antenna_params['phase_center_offset'] ** 2 + antenna_params['lever_arm_length'] ** 2)
        self.reflector_list = reflector_list
        #Antenna pattern in polar coordinates
        ant_bw = np.deg2rad(0.4)#Antenna beamwidth
        tau_ant = ant_bw / self.omega
        self.ant_pat = lambda tau: 10**(30 * (np.sinc(((tau)) / tau_ant)**3)/10) #this function evaluated the antenna pattern at the slow time tau
        #Squint given in slow time delay
        squint_ang = np.deg2rad(self.chirp_freq * self.squint_rate)
        squint_ang = np.deg2rad(gpf.squint_angle(self.chirp_freq, gpf.KU_WIDTH, gpf.KU_DZ_ALT))
        squint_ang = squint_ang - squint_ang[squint_ang.shape[0]/2]
        self.squint_t = squint_ang / self.omega
        #Fast and slow time vectors
        self.fast_time =  np.arange(acquisition_params['CHP_num_samp']) * 1/self.ADC_sample_rate
        self.slow_time =  np.arange(self.nl_tot) * self.chirp_duration
        #Raw parameters
        raw_par = {}
        raw_par['time_start'] = '2016-02-24 10:52:18.670216+00:00'
        raw_par['geographic_coordinates'] = '47.4056316667  8.5055900000  523.2000  47.3000'
        raw_par['RF_center_freq'] = self.f0
        raw_par['RF_freq_min'] = acquisition_params['CHP_freq_min']
        raw_par['RF_freq_max'] = acquisition_params['CHP_freq_max']
        raw_par['RF_chirp_rate'] = self.k
        raw_par['CHP_num_samp'] = acquisition_params['CHP_num_samp']
        raw_par['TX_mode'] = 'TX_RX_SEQ'
        raw_par['TX_RX_SEQ'] = 'AAAl'
        raw_par['ADC_capture_time'] = self.ADC_capture_time
        raw_par['ADC_sample_rate'] = self.ADC_sample_rate
        raw_par['STP_antenna_start'] = acquisition_params['STP_antenna_start']
        raw_par['STP_antenna_end'] = acquisition_params['STP_antenna_end']
        raw_par['STP_gear_ratio'] = acquisition_params['STP_gear_ratio']
        raw_par['TSC_acc_ramp_angle'] = ang_acc
        raw_par['TSC_acc_ramp_time'] = t_acc
        raw_par['TSC_acc_ramp_step'] = acquisition_params['STP_acc_ramp_step']
        raw_par['TSC_rotation_speed'] = acquisition_params['STP_rotation_speed']
        raw_par['antenna_elevation'] = acquisition_params['antenna_elevation']
        raw_par['IMA_atten_dB'] = 48
        gpf.dict_to_par(raw_par, raw_par_file)
        self.raw_file = raw_file


    def fmcw_signal(self, fast_time, r):
        """
        Simulate FMCW signal for a given range and fast time
        Returns
        -------

        """
        sig = np.exp(-1j * 4 * np.pi * r * self.f0 / gpf.C) * \
              np.exp(-1j * np.pi * 4 * r/gpf.C * self.k * fast_time) * \
              np.exp(-1j * 4 * np.pi * (r/gpf.C)**2 * self.k)
        return sig

    def simulate(self):
        """
        Simulate the scan
        """
        sig = np.zeros((self.number_of_chirp_samples + 1, self.nl_tot), dtype=np.complex64)#acquired raw data
        print(sig.shape)
        for idx_fast, t in enumerate(self.fast_time):#iterate over all fast times
            for targ_r, targ_az in self.reflector_list:#for all reflectors
                tau_targ = (targ_az - self.az_min) / self.omega#target location in slow time (angle)
                nu = t + self.slow_time + self.squint_t[idx_fast] #time variable slow time + fast time + time delay caused by squint
                r = distance_from_target(targ_r, self.r_arm, tau_targ, nu, self.omega)
                chirp = self.fmcw_signal(t, r)  * self.ant_pat(nu - tau_targ) #Chirp + antenna pattern
                #Interpolate for squint
                sig[idx_fast, :] += chirp #coherent sum of all signals
        (sig).T.astype(gpf.type_mapping['SHORT INTEGER']).tofile(self.raw_file)
        return sig




    # def correct_squint(self, rawdata, use_linear=False):
    #      for freq, idx_freq in enumerate(self.chirp_freq):
    #         tau_vec = self.slow_time
    #         az_vec_corr = az_vec + squint_t[idx_freq] / t_chirp
    #         signal_corr[idx_freq,:] = np.interp(az_vec, az_vec_corr, signal[idx_freq,:].real)
    #
    # def range_compression(self, kbeta, z):
    #     r_win = sig.kaiser(raw_data.shape[0], kbeta)
    #     zero = np.ones(raw_data.shape[0])
    #     zero[0:z] = np.hamming(2 * z)[0:z]
    #     zero[-z:] = np.hamming(2 * z)[-z:]
    #     fshift = np.ones(raw_data.shape[0]/2 +1)
    #     fshift[0::2] = -1
    #     s_compr = np.fft.rfft(raw_data.imag * r_win[:,None] * zero[:,None], axis =0 ) * fshift[:, None]
    #     return s_compr

#Load real data
real_data = '/data/HIL/20160224/slc_old/20160224_135024_AAAl.slc'
HH_slc, HH_par = gpf.load_dataset(real_data + '.par', real_data)

r_vec = HH_par['near_range_slc'][0]  + HH_par['range_pixel_spacing'][0] * np.arange(HH_par['range_samples'])
az_vec = HH_par['GPRI_az_start_angle'][0]  + HH_par['GPRI_az_angle_step'][0] * np.arange(HH_par['azimuth_lines'])

plt.figure()
plt.imshow(np.abs(HH_slc)**0.2, extent=[r_vec[-1],r_vec[0],az_vec[-1],az_vec[0]][::-1], origin='lower', aspect=1/10.0)

radar_par = '/data/Simulations/gpri_1ms.prf'
HH_ant_par = '/data/Simulations/HH_ant.par'
HH_raw = '/data/Simulations/HH.raw'
HH_raw_par = '/data/Simulations/HH.raw_par'


ref_list = [(350, np.deg2rad(11)),(362,np.deg2rad(10)),(362,np.deg2rad(8)),(590,np.deg2rad(10))]
HH_sim = acquisitionSimulator(radar_par, HH_ant_par, ref_list, HH_raw, HH_raw_par)
HH_sim.simulate()



# HH_rgb, pal, rest = vf.dismph(HH_rc, k=0.2)

# #Simulate the fmcw signal for a given range
# def fmcw_signal(r, t_chirp, chirp_rate, adc_rate, f0):
#     #Number of samples in chirp
#     n_chirp = adc_rate * t_chirp
#     #Fast time vector
#     fast_time = np.arange(n_chirp) * 1/adc_rate
#     #Dechirped signal
#     lam = C / f0
#     #Propagation phase
#     signal = np.exp(-1j * 4 * np.pi * r * f0 / C) * \
#              np.exp(-1j * np.pi * 4 * r/C * chirp_rate * fast_time) * \
#               np.exp(-1j * 4 * np.pi * (r/C)**2 * chirp_rate)
#     return signal

#Simulate distance from a target



# #Combine into scanning signal
# def scanning_signal(t_chirp, chirp_rate, adc_rate, f0, r_c, r_ant, tau_target, omega, nlines):
#     #Number of samples in chirp
#     n_chirp = adc_rate * t_chirp
#     print(n_chirp)
#     #Fast time vector
#     fast_time = np.arange(n_chirp) * 1/adc_rate
#     #Slow time vector
#     slow_time = np.arange(0,n_lines) * t_chirp
#     #Dechirped signal
#     lam = C / f0
#     #Signal vector
#     sig_vector = np.zeros((n_chirp, nlines), dtype=np.complex64)
#     chirp_freq = f0 - 100e6 + chirp_rate * fast_time
#     #Squint vector
#     squint_ang = np.deg2rad(gpf.squint_angle(chirp_freq, gpf.KU_WIDTH, gpf.KU_DZ_ALT))
#     squint_ang = np.deg2rad(chirp_freq * 4.11e-9)
#     squint_ang = squint_ang - squint_ang[squint_ang.shape[0]/2]
#     squint_t = squint_ang / omega
#     plt.show()
#     #Propagation phase
#     for idx_t, t in enumerate(slow_time):
#         tau = fast_time + t
#         r = distance_from_target(r_c, r_ant, tau + squint_t, tau_target, omega)
#         signal = np.exp(-1j * 4 * np.pi * r * f0 / C) * \
#              np.exp(-1j * np.pi * 4 * r/C * chirp_rate * fast_time) * \
#              np.exp(1j * np.pi * 2*  (r/C)**2 * chirp_rate)
#         #Antenna pattern
#         ant_pat = np.sinc(((tau - tau_target + squint_t)) / tau_ant)**3 * 10
#         sig_vector[:, idx_t] = signal * ant_pat
#     return sig_vector
#
#
# def correct_squint(signal, t_chirp, chirp_rate, adc_rate, f0, omega):
#     signal_corr = signal * 1
#     nlines = signal.shape[1]
#     n_chirp = adc_rate * t_chirp
#     #Fast time vector
#     fast_time = np.arange(n_chirp) * 1/adc_rate
#     #Slow time vector
#     slow_time = np.arange(0,n_lines) * t_chirp
#     #Dechirped signal
#     lam = C / f0
#     #Signal vector
#     sig_vector = np.zeros((n_chirp, nlines), dtype=np.complex64)
#     chirp_freq = f0 - bw/2 + chirp_rate * fast_time
#     #Squint vector
#     squint_ang_exact = np.deg2rad(gpf.squint_angle(chirp_freq, gpf.KU_WIDTH, gpf.KU_DZ))
#     squint_ang_exact = squint_ang_exact - squint_ang_exact[squint_ang_exact.shape[0]/2]
#     squint_ang = np.deg2rad(chirp_freq * 3.5e-9)
#     plt.plot(chirp_freq, squint_ang - squint_ang_exact)
#     plt.show()
#     squint_ang = squint_ang - squint_ang[squint_ang.shape[0]/2]
#     squint_t = squint_ang / omega
#     for idx_freq in range(signal.shape[0]):
#         az_vec = np.arange(signal.shape[1])
#         az_vec_corr = az_vec + squint_t[idx_freq] / t_chirp
#         signal_corr[idx_freq,:] = np.interp(az_vec, az_vec_corr, signal[idx_freq,:].real)
#     return signal_corr
#
#
# s = scanning_signal(chirp_length, chirp_rate, adc_rate, f0, r_c, r_arm, target_tau, omega,n_lines)
# s_meas = s.real
# #Correct antenna squint
# s_desq = correct_squint(s_meas, chirp_length, chirp_rate, adc_rate, f0, omega)
#
# r_win = sig.kaiser(s.shape[0], 5.0)
# zero = np.ones(s.shape[0])
# zero[0:z] = np.hamming(2 * z)[0:z]
# zero[-z:] = np.hamming(2 * z)[-z:]
# fshift = np.ones(s.shape[0]/2 +1)
# fshift[0::2] = -1
# s_compr = np.fft.rfft(s_meas * r_win[:,None] * zero[:,None], axis =0 ) * fshift[:, None]
# s_compr_desq = np.fft.rfft(s_desq * r_win[:,None] * zero[:,None], axis =0 ) * fshift[:, None]
# rgb, pal, crap = vf.dismph(s_compr, k=0.1)
# rgb_corr, pal, crap = vf.dismph(s_compr_desq, k=0.3)

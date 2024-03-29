import numpy as np
import pyrat.fileutils.gpri_files as gpf


def distance_from_target(r_arm, r_ph, r_sl, tau, tau_target, omega):
    theta = omega * (tau - tau_target)
    ph, d = cal.distance_from_phase_center(r_arm, r_ph, r_sl, theta)
    return d


#
# def ant_pat(tau, bw):
#

class acquisitionSimulator:
    def __init__(self, prf_file, ant_file, reflector_list, raw_file, raw_par_file):
        acquisition_params = gpf.par_to_dict(prf_file)
        antenna_params = gpf.par_to_dict(ant_file)
        # Parameters relating to chirp and scanning
        self.number_of_chirp_samples = acquisition_params['CHP_num_samp']
        self.chirp_duration = acquisition_params['CHP_num_samp'] / acquisition_params['ADC_sample_rate']
        self.f0 = acquisition_params['RF_center_freq']
        self.bw = acquisition_params['CHP_freq_max'] - acquisition_params['CHP_freq_min']
        self.k = self.bw / self.chirp_duration  # chirp rate in hz/sec
        self.az_min = np.deg2rad(acquisition_params['STP_antenna_start'])
        self.az_max = np.deg2rad(acquisition_params['STP_antenna_end'])
        self.omega = np.deg2rad(acquisition_params['STP_rotation_speed'] * 4)
        self.ADC_capture_time = np.abs(self.az_min - self.az_max) / self.omega  # total slow time
        # Add the capture time due to the ramp
        (ang, rate_max, t_acc, ang_acc, rate_max) = gpf.ant_pos(0.0, acquisition_params['STP_antenna_start'],
                                                                acquisition_params['STP_antenna_end'],
                                                                acquisition_params['STP_gear_ratio'],
                                                                acquisition_params['STP_rotation_speed'])
        self.ADC_capture_time += 2 * t_acc
        self.nl_tot = self.ADC_capture_time / self.chirp_duration  # total number of azimuth lines
        self.ADC_sample_rate = acquisition_params['ADC_sample_rate']
        self.chirp_freq = self.f0 + np.linspace(acquisition_params['CHP_freq_min'], acquisition_params['CHP_freq_max'],
                                                acquisition_params['CHP_num_samp'])
        # Antenna parameters
        self.squint_rate = np.deg2rad(antenna_params['antenna_squint_rate'])
        self.r_arm = antenna_params['lever_arm_length']
        self.r_ph = antenna_params['phase_center_offset']
        # self.r_arm = np.sqrt(antenna_params['phase_center_offset'] ** 2 + antenna_params['lever_arm_length'] ** 2)
        # self.arm_angle = np.arctan2(antenna_params['phase_center_offset'],antenna_params['lever_arm_length'])
        self.reflector_list = reflector_list
        # Antenna pattern in polar coordinates
        ant_bw = np.deg2rad(0.4)  # Antenna beamwidth
        tau_ant = ant_bw / self.omega
        self.ant_pat = lambda tau: 10 ** (1 * (np.sinc(
            tau / float(tau_ant)) ** 4) / 10.0) - 1  # this function evaluated the antenna pattern at the slow time tau
        # Squint given in slow time delay
        squint_ang = (self.chirp_freq * self.squint_rate)
        squint_ang = np.deg2rad(gpf.squint_angle(self.chirp_freq, gpf.KU_WIDTH, gpf.KU_DZ_ALT))
        # squint_ang = squint_ang - squint_ang[squint_ang.shape[0]/2]
        self.squint_t = squint_ang / self.omega
        # Fast and slow time vectors
        self.fast_time = np.arange(acquisition_params['CHP_num_samp']) * 1 / self.ADC_sample_rate
        self.slow_time = np.arange(self.nl_tot) * self.chirp_duration
        # Raw parameters
        raw_par = {'time_start': '2016-02-24 10:52:18.670216+00:00',
                   'geographic_coordinates': '47.4056316667  8.5055900000  523.2000  47.3000',
                   'RF_center_freq': self.f0, 'RF_freq_min': acquisition_params['CHP_freq_min'],
                   'RF_freq_max': acquisition_params['CHP_freq_max'], 'RF_chirp_rate': self.k,
                   'CHP_num_samp': acquisition_params['CHP_num_samp'], 'TX_mode': 'TX_RX_SEQ', 'TX_RX_SEQ': 'AAAl',
                   'ADC_capture_time': self.ADC_capture_time, 'ADC_sample_rate': self.ADC_sample_rate,
                   'STP_antenna_start': acquisition_params['STP_antenna_start'],
                   'STP_antenna_end': acquisition_params['STP_antenna_end'],
                   'STP_gear_ratio': acquisition_params['STP_gear_ratio'], 'TSC_acc_ramp_angle': ang_acc,
                   'TSC_acc_ramp_time': t_acc, 'TSC_acc_ramp_step': acquisition_params['STP_acc_ramp_step'],
                   'TSC_rotation_speed': acquisition_params['STP_rotation_speed'] * 4,
                   'antenna_elevation': acquisition_params['antenna_elevation'], 'IMA_atten_dB': 48}
        gpf.dict_to_par(raw_par, raw_par_file)
        self.raw_file = raw_file

    def fmcw_signal(self, fast_time, r):
        """
        Simulate FMCW signal for a given range and fast time
        Returns
        -------

        """
        sig = np.exp(-1j * 4 * np.pi * r * self.f0 / gpf.C) * \
              np.exp(-1j * np.pi * 4 * r / gpf.C * self.k * fast_time) * \
              np.exp(1j * 4 * np.pi * (r / gpf.C) ** 2 * self.k)
        return sig

    def simulate(self):
        """
        Simulate the scan
        """
        sig = np.zeros((self.number_of_chirp_samples + 1, self.nl_tot), dtype=np.complex64)  # acquired raw data
        print(sig.shape)
        for idx_fast, t in enumerate(self.fast_time):  # iterate over all fast times
            for targ_r, targ_az in self.reflector_list:  # for all reflectors
                tau_targ = (targ_az - self.az_min) / self.omega  # target location in slow time (angle)
                nu = t + self.slow_time - self.squint_t[
                    idx_fast]  # time variable slow time + fast time + time delay caused by squint
                r = distance_from_target(self.r_arm, self.r_ph, targ_r, nu, tau_targ, self.omega)
                chirp = self.fmcw_signal(t, r) * self.ant_pat(nu - tau_targ)  # Chirp + antenna pattern
                # Interpolate for squint
                sig[idx_fast, :] += chirp  # coherent sum of all signals
        # First of all, normalize the signal
        sig /= np.max(np.abs(sig))
        sig *= gpf.TSF
        sig.imag.T.astype(gpf.type_mapping['SHORT INTEGER']).tofile(self.raw_file)
        return sig


radar_par = '/data/Simulations/gpri_1ms.prf'
HH_ant_par = '/data/Simulations/HH_ant.par'
HH_raw = '/data/Simulations/HH.raw'
HH_raw_par = '/data/Simulations/HH.raw_par'
VV_ant_par = '/data/Simulations/VV_ant.par'
VV_raw = '/data/Simulations/VV.raw'
VV_raw_par = '/data/Simulations/VV.raw_par'

ref_list = [(350, np.deg2rad(11)), (362, np.deg2rad(10)), (362, np.deg2rad(8)), (590, np.deg2rad(10))]
ref_list = [(550.8, np.deg2rad(11))]

HH_sim = acquisitionSimulator(radar_par, HH_ant_par, ref_list, HH_raw, HH_raw_par)
HH_sim.simulate()
VV_sim = acquisitionSimulator(radar_par, VV_ant_par, ref_list, VV_raw, VV_raw_par)
vv_sig = VV_sim.simulate()
print('dono')

x = np.linspace(-5, 5, 1000)
xx, yy = np.meshgrid(x, x)
r = np.sqrt(xx ** 2 + yy ** 2)
th = np.arctan2(yy, xx)
pat = HH_sim.ant_pat(th / np.deg2rad(1.2))
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('prf', help='Profile file',type=str)
#     parser.add_argument('ant_par', help='Antenna parameters',type=str)
#     parser.add_argument('raw', help='Raw parameters',type=str)
#     parser.add_argument('raw_par', help='Raw file',type=str)
#     parser.add_argument('raw_sim_par', help='Raw parameters',type=str)
#     parser.add_argument('raw_sim', help='Raw file',type=str)
#     args = parser.parse_args()
#     #Obtain
#     #Simulate
#     sim = acquisitionSimulator(args.prf, args.ant_par, ref_list, args.raw_sim, args.raw_sim_par)
#     rc.gpriRangeProcessor()
# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         pass

import pyrat.gpri_utils.calibration as cal

r_arm = 0.25
r_ph = 0.15
theta = np.deg2rad(np.linspace(-10, 10))
theta_t = np.deg2rad(5)
r_t = 500
angle, dist = cal.distance_from_phase_center(r_arm, r_ph, r_t, theta, theta_t)

t = np.linspace(0, 1)
e = np.exp(1j * 2 * t + 0.25 * 1j)
e1 = np.exp(1j * 2 * t + 0.45 * 1j)

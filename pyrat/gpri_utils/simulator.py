#!/usr/bin/python
__author__ = 'baffelli'
import argparse
import sys

import pyfftw.interfaces.numpy_fft as _fftp

import numpy as _np

from . import calibration as _cal
from ..fileutils import gpri_files as _gpf

import matplotlib.pyplot as plt

# Antenna rotation arm length
r_ant = 0.25
# antenna beamwidth
ant_bw = 0.485


def distance_to_delay(distance):
    return distance / _gpf.C


def if_signal(duration, samp_rate, r_dist, fc, bw):
    delay = distance_to_delay(r_dist)
    n_samp = duration * samp_rate
    t = _np.arange(0, n_samp-1) * 1 / samp_rate
    t, tau = _np.meshgrid(t, delay, indexing='ij')
    chirp_rate = bw * 1 / duration
    # Backscatter signal
    bs_sig = _np.exp(1j * 2 * _np.pi * (fc * (t - tau) + 1 / 2 * chirp_rate * (t - tau) ** 2))
    # Reference chirp
    ref_chirp = _np.exp(1j * 2 * _np.pi * (fc * t + 1 / 2 * chirp_rate * (t) ** 2))
    dechirp = (bs_sig * ref_chirp.conj()).real
    return dechirp


def ant_pat(samples, bw):
    pat = _np.sinc(samples   / bw) ** 2
    return pat



class RadarSimulator:
    def __init__(self, targets, prf, r_ph, squint=True, antenna_bw=0.2, squint_rate=4.2e-9, chan='AAAl'):
        """
        Initializes a gpri raw data simulator

        Parameters
        ----------
        targets : iterable
            List of lists of targets,
            each list contains [distance in meter, angle in degress, rcs]
        prf : pyrat.fileutils.ParameterFile
        r_ph : float
            Antenna horizontal phase center shift
        squint : bool
            If set to true, the antenna squint will be simulated
        Returns
        -------
        pyrat.fileutils.rawData
        """
        self.squint_rate=  squint_rate
        self.bw_mult = 4
        self.antenna_bw = antenna_bw
        self.squint = squint
        self.prf = prf
        self.r_ph = r_ph
        # Block length in range is 1 + number of chirp samples
        self.block_length = self.prf.CHP_num_samp + 1
        # Length of one cycle in seconds
        self.tcycle = self.block_length / self.prf['ADC_sample_rate']
        # Angle per cycle
        self.ang_per_tcycle = self.tcycle * self.prf['STP_rotation_speed']
        # Angle difference
        ang_diff = self.prf['STP_antenna_start'] - self.prf['STP_antenna_end']
        # Number of samples
        self.azimuth_samples = int(abs(ang_diff) / self.ang_per_tcycle)
        # Chirp duration
        self.chirp_duration = self.prf['CHP_num_samp'] / self.prf['ADC_sample_rate']
        # List of targets
        # with open(args.targets) as targ_file:
        #     lines = targ_file.read().splitlines()
        #     targs = [map(float, line.split(' ')) for line in lines]
        # print(targs)
        self.targ_list = targets
        # Bandwidth
        self.bw = self.prf['CHP_freq_max'] - self.prf['CHP_freq_min']
        # prepare raw data dictionary
        raw_dict = _gpf.default_raw_par()
        raw_dict['CHP_num_samp'] = self.prf['CHP_num_samp']
        raw_dict['RF_chirp_rate'] = self.bw / self.tcycle
        raw_dict['ADC_sample_rate'] = self.prf['ADC_sample_rate']
        raw_dict['ADC_capture_time'] = self.tcycle * self.azimuth_samples
        raw_dict['RF_freq_max'] = self.prf['CHP_freq_max']
        raw_dict['RF_center_freq'] = self.prf['RF_center_freq']
        raw_dict['RF_freq_min'] = self.prf['CHP_freq_min']
        raw_dict['STP_rotation_speed'] = self.prf['STP_rotation_speed']
        raw_dict['TSC_rotation_speed'] = self.prf['STP_rotation_speed']
        raw_dict['STP_antenna_start'] = self.prf['STP_antenna_start']
        raw_dict['STP_antenna_end'] = self.prf['STP_antenna_end']
        raw_dict['TSC_acc_ramp_step'] = self.prf['STP_acc_ramp_step']
        raw_dict['TX_RX_SEQ'] = chan
        # raw_dict['TSC_version'] = 'SW V3.04'
        # raw_dict['TX_mode'] = 'None'
        # raw_dict['TX_RX_SEQ'] = 'AAAl'
        # raw_dict['CHP_version'] = 'SW X2.00'
        # raw_dict['IMA_atten_dB'] = 44.0
        # raw_dict['time_start'] = str(_dt.datetime.now())
        # Compute mechanical antenna parameters
        (ang, rate_max, t_acc, ang_acc, rate_max) = _gpf.ant_pos(0.0, self.prf['STP_antenna_start'],
                                                                 self.prf['STP_antenna_end']
                                                                 , self.prf['STP_gear_ratio'],
                                                                 self.prf['STP_rotation_speed'])
        raw_dict['TSC_acc_ramp_angle'] = ang_acc
        raw_dict['TSC_acc_ramp_time'] = t_acc
        raw_dict['geographic_coordinates'] = [0, 0, 0, 0]
        raw_dict['antenna_elevation'] = self.prf['antenna_elevation']
        self.raw_par = raw_dict
        # _gpf.dict_to_par(raw_dict, args.raw_par_out)

    def simulate(self):
        raw_data = _np.zeros([self.block_length, self.azimuth_samples], dtype=_np.float32)
        raw_data = _gpf.rawData(self.raw_par, raw_data, from_array=True)
        #Generate phase cente location using a dummy slc
        raw_data.compute_slc_parameters()
        slc_dict = raw_data.fill_dict()
        slc = _gpf.gammaDataset(slc_dict, raw_data)
        r_ant = _np.linalg.norm(slc.phase_center[0:2])
        # Antenna pattern
        for targ in self.targ_list:
            #For each target, construct slice
            min_pos = -self.bw_mult*self.antenna_bw/2
            max_pos = self.bw_mult*self.antenna_bw/2
            az_slice = _np.deg2rad(_np.arange(min_pos, max_pos, step=self.ang_per_tcycle))

            filt, dist = _cal.distance_from_phase_center(r_ant, self.r_ph, targ[0], az_slice,
                                                         _gpf.lam(self.prf.RF_center_freq), wrap=False)
            #Compute indices of target location
            targ_start = (min_pos - targ[1] - self.prf['STP_antenna_start'])//self.ang_per_tcycle
            targ_stop = (max_pos - targ[1] - self.prf['STP_antenna_start'])//self.ang_per_tcycle
            # Antenna pattern

            ant_pattern = ant_pat(az_slice,_np.deg2rad(self.antenna_bw))
            #Get range location
            pixel_loc = int((targ[0] - slc.near_range_slc)//slc.range_pixel_spacing)
            scale = raw_data.scale[pixel_loc]
            raw_data[1::, targ_start:targ_stop] += 1/scale* 10 ** (targ[2] / 10) * if_signal(self.tcycle, self.prf['ADC_sample_rate']
                                                                    , dist, self.prf['RF_center_freq'],
                                                                    self.bw) * ant_pattern[None, :]
            # Convert to GPRI data
        # Apply scaling
        raw_data =(raw_data).astype(_gpf.type_mapping['SHORT INTEGER'])
        raw_data = _gpf.rawData(self.raw_par, raw_data, from_array=True)
        if self.squint:
            # Apply squint by using correct_squint with the opposite rate
            raw_data = _gpf.correct_squint(raw_data, squint_function=lambda f,w:-_gpf.model_squint(f +self.prf['RF_center_freq']))
            # ang_vec = _np.arange(raw_data.shape[1])
            # # Create frequency vector
            # freq_vec = _np.linspace(self.prf['CHP_freq_min'], self.prf['CHP_freq_max'], self.prf['CHP_num_samp'],
            #                         dtype=float) + \
            #            + self.prf['RF_center_freq']
            # sq_ang = _gpf.linear_squint(freq_vec, self.squint_rate)
            # sq_vec = (sq_ang - sq_ang[sq_ang.shape[0] / 2]) / self.ang_per_tcycle
            # sq_vec = _np.insert(sq_vec, 0, 0)
            # for idx_r in range(0, raw_data.shape[0]):
            #     if idx_r % 100 == 0:
            #         print("interp range: {}, squint in samples {} ".format(idx_r, sq_vec[idx_r]))
            #     az_new = ang_vec + sq_vec[idx_r]
            #     raw_data[idx_r, :] = _np.interp(az_new, ang_vec, raw_data[idx_r, :], left=0.0, right=0.0)

        return raw_data


def main():
    # Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('prf',
                        help="Radar acquisition profile")
    parser.add_argument('targets',
                        help="List of targets each line containing azimuth and range indices")
    parser.add_argument('raw_out', type=str,
                        help="Output simulated raw dataset")
    parser.add_argument('raw_par_out', type=str,
                        help="Output raw parameters")
    parser.add_argument('--r_ph',
                        help="Antenna phase center horizontal displacement",
                        type=float, default=0.12)
    parser.add_argument('--squint',
                        help="Apply frequency variable antenna squint"
                        , default=False, action='store_true', dest='squint')
    # Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    sim = radarSimulator(args)
    raw_data = sim.simulate()
    # Convert to int 16
    raw_data = _np.int16(raw_data * 32768)
    with open(args.raw_out, 'wb') as of:
        raw_data.T.astype(_gpf.type_mapping['SHORT INTEGER']).tofile(of)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

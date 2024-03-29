#!/usr/bin/python
__author__ = 'baffelli'
C = 299792458.0  # speed of light m/s
KU_WIDTH = 15.798e-3  # WG-62 Ku-Band waveguide width dimension
KU_DZ = 10.682e-3  # Ku-Band Waveguide slot spacing
RANGE_OFFSET = 3

import argparse
import sys

import numpy as _np
import pyrat.fileutils.gpri_files as _gpf
import pyrat.gpri_utils.calibration as _cal
import scipy.signal as _sig


def first_derivative(r_arm, r_ph, r_sl):
    r_ant = _np.sqrt(r_arm ** 2 + r_ph ** 2)
    denom = r_sl
    alpha = _np.arctan(r_arm / r_ph)
    num = 2 * r_ant * r_sl * _np.cos(alpha)
    return num / denom


def second_derivative(r_arm, r_ph, r_sl):
    r_ant = _np.sqrt(r_arm ** 2 + r_ph ** 2)
    denom = r_sl ** (3 / 2) * 4
    num = r_ant ** 2 * 2 * r_sl - first_derivative(r_arm, r_ph, r_sl)
    return num / denom


class gpriAzimuthProcessor:
    def __init__(self, args):
        # Load slc
        self.slc = _gpf.gammaDataset(args.slc_par, args.slc).astype(_np.complex64)
        self.args = args
        # Rotation arm is computed from the parameter file if not specified
        if not hasattr(args, 'r_ant'):
            self.r_ant = _gpf.xoff + _np.cos(_np.deg2rad(self.slc.GPRI_ant_elev_angle[0])) * _gpf.ant_radius
        else:
            self.r_ant = args.r_ant

    def correct2d(self):
        # Construct range vector
        r_vec = self.slc.near_range_slc[0] + _np.arange(self.slc.shape[0]) * self.slc.range_pixel_spacing[0]
        # Azimuth vector for the entire image
        az_vec_image = _np.deg2rad(self.slc.GPRI_az_start_angle[0]) + _np.arange(self.slc.shape[1]) * _np.deg2rad(
            self.slc.GPRI_az_angle_step[0])
        # Compute integration window size in samples
        ws_samp = int(self.args.ws / self.slc.GPRI_az_angle_step[0])
        # Filtered slc
        if not self.args.discard_samples:
            slc_filt = _np.zeros(self.slc.shape, dtype=_np.complex64)
        # Integration window
        theta = _np.arange(-ws_samp / 2, ws_samp / 2) * _np.deg2rad(self.slc.GPRI_az_angle_step[0])
        for idx_r, r_sl in enumerate(r_vec):
            # print(idx_r)
            for idx_az, az in enumerate(az_vec_image[ws_samp::]):
                theta = theta + az
                idx_az_real = idx_az + ws_samp / 2
                filt, dist = _cal.distance_from_phase_center(self.r_ant, self.args.r_ph, r_sl, theta[1:] - az,
                                                             wrap=False)
                filt = _np.exp(-1j * filt)
                slc_filt[idx_r, idx_az] = _np.sum(
                    filt.conj() * self.slc[idx_r, idx_az_real - int(ws_samp / 2.0):idx_az_real + int(ws_samp / 2.0)])
        return slc_filt

    def correct(self):
        # Construct range vector
        r_vec = self.slc.near_range_slc[0] + _np.arange(self.slc.shape[0]) * self.slc.range_pixel_spacing[0]
        # Compute integration window size in samples
        ws_samp = int(self.args.ws / self.slc.GPRI_az_angle_step[0])
        # Filtered slc has different sizes depending
        # if we keep all the samples after filtering
        if not self.args.discard_samples:
            slc_filt = self.slc * 1
        else:
            slc_filt = self.slc[:, ::ws_samp] * 1
        # process each range line
        theta = _np.arange(-ws_samp / 2, ws_samp / 2) * _np.deg2rad(self.slc.GPRI_az_angle_step[0])
        for idx_r, r_sl in enumerate(r_vec):
            filt, dist = _cal.distance_from_phase_center(self.r_ant, self.args.r_ph, r_sl, theta, wrap=False)
            lam = _gpf.C / 17.2e9
            # Normal matched filter
            matched_filter = _np.exp(-1j * filt) * _np.exp(-1j * 4 * _np.pi * r_sl / lam)
            filter_output = _sig.fftconvolve(self.slc[idx_r, :], matched_filter, mode='same')
            if self.args.discard_samples:
                filter_output = filter_output[::ws_samp]
                slc_filt.GPRI_az_angle_step[0] = self.slc.GPRI_az_angle_step[0] * ws_samp
                slc_filt.azimuth_lines = filter_output.shape[0]
            else:
                pass
            slc_filt[idx_r, :] = filter_output
            if idx_r % 1000 == 0:
                print('Processing range index: ' + str(idx_r))
        return slc_filt


def main():
    # Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('slc',
                        help="Uncorrected slc file")
    parser.add_argument('slc_par',
                        help="Slc parameters")
    parser.add_argument('slc_out', type=str,
                        help="Output slc")
    parser.add_argument('slc_par_out', type=str,
                        help="Output slc parameters")
    parser.add_argument('--r_ant', type=float, default=argparse.SUPPRESS,
                        help="Antenna rotation arm length. If not specified, it is calculated from the GPRI parameter file")
    parser.add_argument('--r_ph',
                        help="Antenna phase center horizontal displacement",
                        type=float, default=0.15)
    parser.add_argument('-w', '--win_size', dest='ws',
                        help="Convolution window size in degrees",
                        type=float, default=0.4)
    parser.add_argument('-d', '--discard-samples', dest='discard_samples',
                        help="Discard samples after convolution",
                        action='store_true')
    # Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    # Create processor object
    proc = gpriAzimuthProcessor(args)
    slc_corr = proc.correct()
    with open(args.slc_out, 'wb') as of:
        slc_corr.T.astype(_gpf.type_mapping['FCOMPLEX']).tofile(of)
    _gpf.dict_to_par(_gpf.par_to_dict(args.slc_par), args.slc_par_out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

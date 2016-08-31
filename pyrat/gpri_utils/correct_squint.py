#!/usr/bin/python
__author__ = 'baffelli'
RANGE_OFFSET = 3

import argparse
import shutil as _shutil
import sys

import numpy as _np
import pyrat.fileutils.gpri_files as _gpf


class squintCorrector:
    def __init__(self, args, grp):
        self.raw_par = grp
        self.args = args

    def correct_squint(self, raw_channel, squint_function):
        # We require a function to compute the squint angle
        squint_vec = squint_function(self.raw_par.freq_vec)
        squint_vec = squint_vec / self.raw_par.ang_per_tcycle
        squint_vec = squint_vec - squint_vec[self.raw_par.freq_vec.shape[0] / 2]
        # In addition, we correct for the beam motion during the chirp
        rotation_squint = _np.linspace(0, self.raw_par.tcycle,
                                       self.raw_par.nsamp) * self.raw_par.grp.TSC_rotation_speed / self.raw_par.ang_per_tcycle
        # Normal angle vector
        angle_vec = _np.arange(raw_channel.shape[1])
        # We do not correct the squint of the first sample
        squint_vec = _np.insert(squint_vec, 0, 0)
        rotation_squint = _np.insert(rotation_squint, 0, 0)
        # Interpolated raw channel
        raw_channel_interp = _np.zeros_like(raw_channel)
        for idx in range(0, raw_channel.shape[0]):
            az_new = angle_vec + squint_vec[idx] - rotation_squint[idx]
            if idx % 500 == 0:
                print_str = "interp sample: {idx}, ,shift: {sh}".format(idx=idx, sh=az_new[0] - angle_vec[0])
                print(print_str)
            raw_channel_interp[idx, :] = _np.interp(az_new, angle_vec, raw_channel[idx, :], left=0.0, right=0.0)
        return raw_channel_interp


# These function take a vector of frequencies and
# return a vector of squint angles in degrees
def model_squint(freq_vec):
    return _gpf.squint_angle(freq_vec, _gpf.KU_WIDTH, _gpf.KU_DZ, k=1.0 / 2.0)


def linear_squint(freq_vec, sq_parameters):
    return _np.polynomial.polynomial.polyval(freq_vec, sq_parameters)


def main():
    # Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('raw',
                        help="GPRI raw file")
    parser.add_argument('raw_par',
                        help="GPRI raw file parameters")
    parser.add_argument('raw_out', type=str,
                        help="Corrected GPRI raw file")
    parser.add_argument('raw_par_out', type=str,
                        help="Parameters of the corrected GPRI raw file")
    parser.add_argument('sq_rate', help='Squint rate in deg/Hz', type=float)
    parser.add_argument('center_squint', help='Squint at the center frequency in degrees', type=float)
    parser.add_argument('--exact', help='If set, uses an exact model for the squint', action='store_true')

    # Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    print(args.sq_rate)

    # Read raw dataqset and parameters
    raw_dict = _gpf.par_to_dict(args.raw_par)
    # Compute parameters
    raw_par = _gpf.rawParameters(raw_dict, args.raw)
    raw_data = _np.fromfile(args.raw, dtype=raw_par.dt).reshape([raw_par.nl_tot,
                                                                 raw_par.block_length]).T
    raw_data = _gpf.load_segment(args.raw,
                                 (raw_par.block_length, raw_par.nl_tot)
                                 , 0, raw_par.block_length, 0,
                                 raw_par.nl_tot,
                                 dtype=_gpf.type_mapping['SHORT INTEGER'])

    # Create squint corrector object
    squint_processor = squintCorrector(args, raw_par)
    if args.exact:
        sf = model_squint
    else:
        # Squint correction function
        sf = lambda freq_vec: linear_squint(freq_vec, [args.center_squint, args.sq_rate])
    # Call processor
    raw_data_corr = squint_processor.correct_squint(raw_data, sf)
    # Write dataset
    with open(args.raw_out, 'wb') as of:
        raw_data_corr.T.astype(_gpf.type_mapping['SHORT INTEGER']).tofile(of)
    # Copy parameters
    _shutil.copyfile(args.raw_par, args.raw_par_out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

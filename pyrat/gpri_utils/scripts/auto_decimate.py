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


class gpriDecimator:
    def __init__(self, args):
        self.slc = _gpf.gammaDataset(args.slc_par, args.slc)
        # Compute decimated sample spacing
        self.dec = args.dec

    def decimate(self):
        arr_dec = _np.zeros((self.slc.shape[0], int(self.slc.shape[1] / self.dec)), dtype=_np.complex64)
        # For each azimuth
        for idx_az in range(arr_dec.shape[1]):
            # Decimated pulse
            dec_pulse = _np.zeros(self.slc.shape[0] * 2 - 2, dtype=_np.float32)
            for idx_dec in range(self.dec):
                current_idx = idx_az * self.dec + idx_dec
                if current_idx % 1000 == 0:
                    print('decimating line: ' + str(current_idx))
                dec_pulse += _np.fft.irfft(self.slc[:, current_idx])
            arr_dec[:, idx_az] = _np.fft.rfft(dec_pulse)
        return arr_dec

    def decimate_by_discarding(self):
        # arr_dec = _np.zeros((self.slc.shape[0], int(self.slc.shape[1]/self.dec)) ,dtype=_np.complex64)
        arr_dec = self.slc[:, ::self.dec]
        return arr_dec


def main():
    # Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('slc', type=str,
                        help="SLC channel file")
    parser.add_argument('slc_par', type=str,
                        help="SLC channel file parameters")
    parser.add_argument('slc_out',
                        help="Decimated slc", type=str)
    parser.add_argument('slc_par_out', type=str,
                        help="Output slc parameters")
    parser.add_argument('--sample_spacing', type=float, default=0.1,
                        help="Decimated slc sample spacing in degrees")
    parser.add_argument('mode', type=int, default=0,
                        help='Decimation mode: 0 (default) adding  n samples, 1 discarding every nth-sample')
    # Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    # Create processor object
    proc = gpriDecimator(args)
    if args.mode == 0:
        slc_dec = proc.decimate()
    else:
        slc_dec = proc.decimate_by_discarding()
    slc_par = _gpf.par_to_dict(args.slc_par)
    slc_par['GPRI_az_angle_step'][0] = slc_par['GPRI_az_angle_step'][0] * args.dec
    slc_par['azimuth_lines'] = slc_dec.shape[1]
    with open(args.slc_out, 'wb') as of:
        slc_dec.T.astype(_gpf.type_mapping['FCOMPLEX']).tofile(of)
    _gpf.dict_to_par(slc_par, args.slc_par_out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

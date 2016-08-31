#!/usr/bin/python
__author__ = 'baffelli'
C = 299792458.0  # speed of light m/s
KU_WIDTH = 15.798e-3  # WG-62 Ku-Band waveguide width dimension
KU_DZ = 10.682e-3  # Ku-Band Waveguide slot spacing
RANGE_OFFSET = 3

import argparse
import sys

import pyrat.fileutils.gpri_files as _gpf


class gpriDecimator:
    def __init__(self, args):
        self.slc = _gpf.gammaDataset(args.slc_par, args.slc)
        self.dec = args.dec

    def decimate(self):
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
    parser.add_argument('dec', type=int,
                        help="Slc decimation factor")
    # Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    # Create processor object
    proc = gpriDecimator(args)
    slc_dec = proc.decimate()
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

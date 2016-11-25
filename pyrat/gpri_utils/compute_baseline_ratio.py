#!/usr/bin/python
__author__ = 'baffelli'
C = 299792458.0  # speed of light m/s
KU_WIDTH = 15.798e-3  # WG-62 Ku-Band waveguide width dimension
KU_DZ = 10.682e-3  # Ku-Band Waveguide slot spacing
RANGE_OFFSET = 3

import argparse
import sys

import pyrat.fileutils.gpri_files as _gpf


def main():
    # Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('par', type=str, nargs=2,
                        help="SLC parameters files")
    # Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    # Load parameters
    par1 = _gpf.par_to_dict(args.par[0])
    par2 = _gpf.par_to_dict(args.par[1])
    ph_center_1 = (par1['GPRI_TX_antenna_position'][0] + par1['GPRI_RX_antenna_position'][0]) / 2.0
    ph_center_2 = (par2['GPRI_TX_antenna_position'][0] + par2['GPRI_RX_antenna_position'][0]) / 2.0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

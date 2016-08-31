#!/usr/bin/python
__author__ = 'baffelli'

import argparse
import sys

import pyrat.fileutils.gpri_files as _gpf


def main():
    # Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('par', type=str, nargs=2,
                        help="SLC parameters file")
    parser.add_argument('base_file', type=str, help='DIFF parameter file')
    # Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    # Load parameters
    par1 = _gpf.par_to_dict(args.par[0])
    par2 = _gpf.par_to_dict(args.par[1])
    # Compute z-phase center position
    ph_center_1 = _gpf.compute_phase_center(par1)
    ph_center_2 = _gpf.compute_phase_center(par2)
    base_dict = _gpf.par_to_dict(args.base_file)
    base_dict['antenna_separation'] = [ph_center_1 - ph_center_2, 'm']
    receiver_1 = _gpf.extract_channel_number(par1['title'][-1])
    receiver_2 = _gpf.extract_channel_number(par2['title'][-1])
    str1 = "GPRI_rx{}_coord".format(receiver_1)
    str2 = "GPRI_rx{}_coord".format(receiver_2)
    base_dict['image_1_transmitter_location'] = par1['GPRI_tx_coord']
    base_dict['image_2_transmitter_location'] = par2['GPRI_tx_coord']
    base_dict['image_1_receiver_location'] = par1[str1]
    base_dict['image_2_receiver_location'] = par2[str2]
    _gpf.dict_to_par(base_dict, args.base_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

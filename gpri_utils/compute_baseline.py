#!/usr/bin/python
__author__ = 'baffelli'

import sys, os
import argparse
import pyrat.fileutils.gpri_files as _gpf
from collections import OrderedDict as _od



def main():
    #Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('par', type=str, nargs=2,
                help="SLC parameters file")
    parser.add_argument('base_file', type=str, help='Baseline parameter file')
    #Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    #Load parameters
    par1 = _gpf.par_to_dict(args.par[0])
    par2 = _gpf.par_to_dict(args.par[1])
    rx_number = lambda par : 1 if par['GPRI_RX_antenna'][1] == 'l' else 2
    #Compute phase center position
    ph_center_1 = (par1['GPRI_tx_coord'][2] + par1['GPRI_rx{num}_coord'.format(num=rx_number(par1))][2])/2.0
    ph_center_2 = (par2['GPRI_tx_coord'][2] + par2['GPRI_rx{num}_coord'.format(num=rx_number(par2))][2])/2.0
    print(ph_center_1)
    print(ph_center_2)
    base_dict = _od()
    base_dict['antenna_separation'] = [ph_center_1 - ph_center_2, 'm']
    _gpf.dict_to_par(base_dict, args.base_file)
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

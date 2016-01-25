#!/usr/bin/python
__author__ = 'baffelli'


import sys, os
import numpy as _np
import argparse
import pyrat.fileutils.gpri_files as _gpf
import osgeo.gdal as _gdal




def main():
    #Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dem', type=str,
                help="Path to the Geotiff DEM")
    parser.add_argument('output_dem', type=str,
                help="Path to the Gamma dem (output)")
    parser.add_argument('output_dem_par', type=str,
                help="Path to the Gamma dem parameter file (output)")
    #Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    #Open the data set
    DS = _gdal.Open(args.dem)




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

#!/usr/bin/python

__author__ = 'baffelli'

import argparse
import sys

import osgeo.gdal as _gdal
import pyrat.fileutils.gpri_files as _gpf
import pyrat.geo.geofun as _geo


def main():
    # Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dem', type=str,
                        help="Path to the Geotiff DEM")
    parser.add_argument('output_dem', type=str,
                        help="Path to the Gamma dem (output)")
    parser.add_argument('output_dem_par', type=str,
                        help="Path to the Gamma dem parameter file (output)")
    # Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    # Open the data set
    DS = _gdal.Open(args.dem)
    # Convert
    dem_dic = _geo.gdal_to_dict(DS)
    _gpf.dict_to_par(dem_dic, args.output_dem_par)
    dem = DS.ReadAsArray()
    dem.astype(_gpf.type_mapping[dem_dic['data_format']]).tofile(args.output_dem)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

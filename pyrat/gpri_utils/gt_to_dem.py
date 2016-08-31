#!/usr/bin/python
import argparse
import sys

from osgeo import gdal


def main():
    # Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dem_gt',
                        help="DEM geotiff path")
    parser.add_argument('dem_par',
                        help="DEM parameters")
    parser.add_argument('dem', type=str,
                        help="Output dem binary")
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    print('Working')
    gt = gdal.Open(args.dem_gt)
    _gpf.geotif_to_dem(gt, args.dem_par, args.dem)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

#!/usr/bin/python

import sys, os
import numpy as _np
import argparse
import scipy as _sp
import scipy.signal as _sig
sys.path.append(os.path.expanduser('~/PhD/trunk/Code/'))
import pyrat.fileutils.gpri_files as _gpf
import pyrat.gpri_utils.calibration as _cal
from collections import namedtuple as _nt
import scipy.signal as _sig
import scipy.ndimage as _ndim
import matplotlib.pyplot as _plt
import pyrat.visualization.visfun as _vf



#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("gc_map", help="Lookup table containing pairs of real valued coordinates", type=str)
parser.add_argument("width", help="The width of the lookup table", type=int)
parser.add_argument("dem_par", help="Dem parameter for the corresponding LUT", type=str)
parser.add_argument("idx", help="Indices to read from the LUT", type=int, nargs=2)
args = parser.parse_args()


#Get shape
shape = _gpf.get_image_size(args.gc_map, args.width, 'FCOMPLEX')
LUT = _np.fromfile(args.gc_map, dtype= _gpf.type_mapping['FCOMPLEX']).reshape(shape[::-1]).T
dem_par = _gpf.par_to_dict(args.dem_par)
#Read coordinates
coord = LUT[args.idx[0], args.idx[1]]
#Convert into geographical coord
x = dem_par['corner_east'][0] + _np.real(coord)*dem_par['post_east'][0]
y = dem_par['corner_north'][0] + _np.imag(coord)*dem_par['post_north'][0]



print("{} {}".format(x,y))

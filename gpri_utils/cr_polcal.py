#!/usr/bin/python
"""
This is a shell wrapper for the calibration function using a single TCR
according to the method in
http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7012094
"""

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
parser = argparse.ArgumentParser(description='Determine calibration parameters using a single TCR.')
parser.add_argument("HHHH", help="The path of the HHHH channel [FCOMPLEX]", type=str)
parser.add_argument("HHVV", help="The path of the HHVV channel [FCOMPLEX]", type=str)
parser.add_argument("VVVV", help="The path of the VVVV channel [FCOMPLEX]", type=str)
parser.add_argument("HVHV", help="The path of the HVHV channel [FCOMPLEX]", type=str)
parser.add_argument("VHVH", help="The path of the VHVH channel [FCOMPLEX]", type=str)
parser.add_argument("HVHH", help="The path of the HVVH channel [FCOMPLEX]", type=str)
parser.add_argument("width", help="Width of image", type=int)
parser.add_argument('refr', help='Range position of the TCR',
                    type=int, default=1)
parser.add_argument('refaz', help='Azimuth position of the TCR',
                    type=int, default=1)
parser.add_argument("f_out", help="The path where to save the f calibration parameter [string]", type=str)
parser.add_argument("g_out", help="The path where to save the g calibration parameter [string]", type=str)
args = parser.parse_args()

#Get shape
shape = _gpf.get_image_size(args.HHHH, args.width, 'FCOMPLEX')
print(shape)


image = _np.fromfile(args.image, dtype= _gpf.type_mapping['FCOMPLEX']).reshape(shape[::-1]).T

correction = _np.exp(1j*phase * args.baseline_ratio)

correction = correction / _np.abs(correction)

out_image = image * correction

with open(args.out, 'wb+') as of:
    out_image.astype(_gpf.type_mapping['FCOMPLEX']).T.tofile(of)

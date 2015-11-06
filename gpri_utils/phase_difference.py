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
parser = argparse.ArgumentParser(description='Compute the phase difference of two fcomplex files')
parser.add_argument("image_1", help="The path of the first image [FCOMPLEX]", type=str)
parser.add_argument("image_2", help="The path of the second image [FCOMPLEX]", type=str)
parser.add_argument("width", help="Width of image", type=float)
parser.add_argument("out", help="Phase difference image", type=str)
args = parser.parse_args()


#Get shape
shape = _gpf.get_image_size(args.image_1, args.width, 'FCOMPLEX')
image_1 = _np.fromfile(args.image_1, dtype= _gpf.type_mapping['FCOMPLEX']).reshape(shape[::-1]).T
image_2 = _np.fromfile(args.image_2, dtype= _gpf.type_mapping['FCOMPLEX']).reshape(shape[::-1]).T


out_image = image_1 * image_2.conj()

with open(args.out, 'wb+') as of:
    out_image.astype(_gpf.type_mapping['FCOMPLEX']).T.tofile(of)

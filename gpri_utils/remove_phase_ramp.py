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
parser.add_argument("image", help="The path of the interferogram to flatten [FCOMPLEX]", type=str)
parser.add_argument("phase", help="The path of the phase to subtract [FCOMPLEX]", type=str)
parser.add_argument("width", help="Width of image", type=int)
parser.add_argument('baseline_ratio', help='The ratio of the baseline, used to rescale the phase',
                    type=float, default=0.5)
parser.add_argument("out", help="The path of the interferogram to flatten [FCOMPLEX]", type=str)
args = parser.parse_args()


#Get shape
shape = _gpf.get_image_size(args.image, args.width, 'FCOMPLEX')
image = _np.fromfile(args.image, dtype= _gpf.type_mapping['FCOMPLEX']).reshape(shape[::-1]).T
phase = _np.fromfile(args.phase, dtype= _gpf.type_mapping['FCOMPLEX']).reshape(shape[::-1]).T


out_image = image * _np.exp(-1j*_np.angle(phase) * args.baseline_ratio)

with open(args.out, 'wb+') as of:
    out_image.astype(_gpf.type_mapping['FCOMPLEX']).T.tofile(of)

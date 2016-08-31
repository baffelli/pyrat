#!/usr/bin/python

import argparse

import numpy as _np
import pyrat.fileutils.gpri_files as _gpf

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("phase", help="The path to the unwrapped phase", type=str)
parser.add_argument("width", help="Width of image", type=int)
parser.add_argument('meas_baseline', help='The baseline where the unwrapped interferogram was measured',
                    type=float, default=1)
parser.add_argument('sim_baseline', help='The baseline to rescale the interferogram to',
                    type=float, default=1)
parser.add_argument("out", help="The path of the output interferogram", type=str)
args = parser.parse_args()

# Get shape
shape = _gpf.get_image_size(args.image, args.width, 'FCOMPLEX')
phase = _np.fromfile(args.phase, dtype=_gpf.type_mapping['FCOMPLEX']).reshape(shape[::-1]).T

correction = _np.exp(-1j * phase * args.sim_baseline / args.meas_baseline)

correction = correction / _np.abs(correction)

with open(args.out, 'wb+') as of:
    correction.astype(_gpf.type_mapping['FCOMPLEX']).T.tofile(of)

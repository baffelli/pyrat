#!/usr/bin/python

import argparse

import numpy as _np
import pyrat.fileutils.gpri_files as _gpf

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("image", help="The path of the interferogram to flatten [FCOMPLEX]", type=str)
parser.add_argument("phase", help="The path of the phase to subtract [Float or FCOMPLEX]", type=str)
parser.add_argument("width", help="Width of image", type=int)
parser.add_argument('original_baseline', help='The ba',
                    type=float, default=1)
parser.add_argument("out", help="The path of the interferogram to flatten [FCOMPLEX]", type=str)
parser.add_argument("type", help='The type of data', choices=['FLOAT', 'FCOMPLEX'])
args = parser.parse_args()

# Get shape
shape = _gpf.get_image_size(args.image, args.width, 'FCOMPLEX')
print(shape)
image = _np.fromfile(args.image, dtype=_gpf.type_mapping['FCOMPLEX']).reshape(shape[::-1]).T
phase = _np.fromfile(args.phase, dtype=_gpf.type_mapping[args.type]).reshape(shape[::-1]).T

if args.type is 'FCOMPLEX':
    phase = _np.angle(phase)
else:
    pass

correction = _np.exp(1j * phase * args.baseline_ratio)

correction = correction / _np.abs(correction)

out_image = image * correction

with open(args.out, 'wb+') as of:
    out_image.astype(_gpf.type_mapping['FCOMPLEX']).T.tofile(of)

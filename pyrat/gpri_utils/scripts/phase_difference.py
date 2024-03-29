#!/usr/bin/python

import argparse

import numpy as _np
import pyrat.fileutils.gpri_files as _gpf

# Argument parser
parser = argparse.ArgumentParser(description='Compute the phase difference of two fcomplex files')
parser.add_argument("image_1", help="The path of the first image [FCOMPLEX]", type=str)
parser.add_argument("image_2", help="The path of the second image [FCOMPLEX]", type=str)
parser.add_argument("width", help="Width of image", type=float)
parser.add_argument("out", help="Phase difference image", type=str)
args = parser.parse_args()

# Get shape
shape = _gpf.get_image_size(args.image_1, args.width, 'FCOMPLEX')
image_1 = _np.fromfile(args.image_1, dtype=_gpf.type_mapping['FCOMPLEX']).reshape(shape[::-1]).T
image_2 = _np.fromfile(args.image_2, dtype=_gpf.type_mapping['FCOMPLEX']).reshape(shape[::-1]).T

out_image = image_1 * image_2.conj()

with open(args.out, 'wb+') as of:
    out_image.astype(_gpf.type_mapping['FCOMPLEX']).T.tofile(of)

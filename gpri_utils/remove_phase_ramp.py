#!/usr/bin/python
import sys
import os
import argparse
sys.path.append(os.path.expanduser('~/PhD/trunk/Code/'))
import pyrat
from pyrat import visfun as vf
from pyrat.gpri_utils import calibration
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, osr
import itertools



#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("par_path", help="The path of the slc that you want to process")
parser.add_argument("slc_path", help="The path of the parameters that you want to process")
parser.add_argument("out_id", help="The id of the output data")
parser.add_argument("azslp", help="Azimuth slope (degrees/sample)", type=float, default=5)
parser.add_argument("rslp", help="Range slope (degrees/sample)", type=float, default=1)
args = parser.parse_args()


#Load slcs
SLC = pyrat.gammaDataset(args.par_path, args.slc_path).astype(np.complex64)

r_vec = np.arange(0,SLC.shape[0])
az_vec = np.arange(0,SLC.shape[1])
xx, yy = np.meshgrid(r_vec, az_vec, indexing='ij')
ramp = np.exp(1j * (np.deg2rad(args.azslp) * yy + np.deg2rad(args.rslp) * xx))

SLC_corr = SLC * ramp

with open(args.out_id, 'wb+') as of:
    SLC_corr.astype(pyrat.type_mapping['FCOMPLEX']).T.tofile(of)

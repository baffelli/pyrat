#!/usr/bin/python

import sys, os
import numpy as _np
import argparse
import scipy as _sp
import scipy.signal as _sig
import pyrat.fileutils.gpri_files as _gpf
import pyrat.gpri_utils.calibration as _cal
from collections import namedtuple as _nt
import scipy.signal as _sig
import scipy.ndimage as _ndim
import matplotlib.pyplot as _plt
import pyrat.visualization.visfun as _vf



#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("HHVV_flat", help="HHVV phase with topography removed", type=str)
parser.add_argument("HH_pwr", help="HH power (mli)", type=str)
parser.add_argument("VV_pwr", help="VV power (mli)", type=str)
args = parser.parse_args()
#Load all channels
chan_dict = {}
chan_dict.fromkeys(['HH','HV','VH','VV'],[[],[],[],[]])
#Create a dictionary with key _ > channel name, value: list with channel and parameter dict
for chan_name in chan_dict.iterkeys():
    chan_dict[chan_name].append( _gpf.gammaDataset(getattr(args,chan_name + '_par'),getattr(args,chan_name)))
    chan_dict[chan_name].append(_gpf.par_to_dict(getattr(args,chan_name + '_par')))

#Compute the baseline for the polarimetric measurements
pol_baseline = _gpf.compute_phase_center(chan_dict['HH'][1]) - _gpf.compute_phase_center(chan_dict['VV'][1])
topo_baseline = _gpf.par_to_dict(args.topo_base)['antenna_separation'][0]

#Compute the HH-VV phase difference and remove the topographic phase
HHVV_phase =chan_dict['HH'][0] * chan_dict['VV'][0].conj() * _np.exp(-1j * pol_baseline / topo_baseline * )

with open(args.out, 'wb+') as of:
    correction.astype(_gpf.type_mapping['FCOMPLEX']).T.tofile(of)

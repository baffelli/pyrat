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
from collections import OrderedDict as _od

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("HHVV_flat", help="HHVV phase with topography removed (fcomplex)", type=str)
parser.add_argument("HHVV_flat_par", help="Parameters of HHVV phase", type=str)
parser.add_argument("HH_pwr", help="HH power (mli)", type=str)
parser.add_argument("HH_pwr_par", help="HH power parameter file (mli)", type=str)
parser.add_argument("VV_pwr", help="VV power (mli)", type=str)
parser.add_argument("VV_pwr_par", help="VV power parameter file (mli)", type=str)
parser.add_argument("HV_pwr", help="HV power (mli)", type=str)
parser.add_argument("HV_pwr_par", help="HV power parameter file (mli)", type=str)
parser.add_argument("VH_pwr", help="VH power (mli)", type=str)
parser.add_argument("VH_pwr_par", help="VH power parameter file (mli)", type=str)
parser.add_argument("ridx", help="Reference reflector range index", type=int)
parser.add_argument("azidx", help="Reference reflector azimuth index", type=int)
parser.add_argument("cal_parameters", help="Computed polarimetric calibration parameters", type=str)
args = parser.parse_args()

# Determine the HH-VV phase imbalance
hhvv_phase, hh_vv_par = _gpf.load_dataset(args.HHVV_flat_par, args.HHVV_flat)
HH_pwr, HH_pwr_par = _gpf.load_dataset(args.HH_pwr_par, args.HH_pwr)
VV_pwr, VV_pwr_par = _gpf.load_dataset(args.VV_pwr_par, args.VV_pwr)
reflector_imbalance_phase = _np.angle(hhvv_phase[args.ridx, args.azidx])
reflector_imbalance_amplitude = _np.sqrt(HH_pwr[args.ridx, args.azidx] / VV_pwr[args.ridx, args.azidx])

#Parameter dict
cal_dict = _od()
cal_dict['HHVV_phase_imbalance'] = [reflector_imbalance_phase, 'rad']
cal_dict['HHVV_amplitude_imbalance'] = reflector_imbalance_amplitude
cal_num = reflector_imbalance_amplitude * _np.exp(1j * reflector_imbalance_phase)
cal_dict['HHVV_amplitude_imbalance_real'] = _np.real(reflector_imbalance_amplitude)
cal_dict['HHVV_amplitude_imbalance_imaginary'] = _np.imag(reflector_imbalance_amplitude)

#Write calibrations parameters
_gpf.dict_to_par(cal_dict, args.cal_parameters)

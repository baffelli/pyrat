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
import pyrat.core.matrices as _mat
# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('HH', help='HH slc', type=str)
parser.add_argument('HH_par', help='HH slc parameters', type=str)
parser.add_argument('HV', help='HV slc', type=str)
parser.add_argument('HV_par', help='HV slc parameters', type=str)
parser.add_argument('VH', help='HV slc', type=str)
parser.add_argument('VH_par', help='VH slc parameters', type=str)
parser.add_argument('VV', help='VV slc', type=str)
parser.add_argument('VV_par', help='VV slc parameters', type=str)
parser.add_argument("topo_phase", help="Topographic phase (unwrapped float)", type=str)
parser.add_argument("topo_phase_par", help="Parameters of topographic phase", type=str)
parser.add_argument("ridx", help="Reference reflector range index", type=int)
parser.add_argument("azidx", help="Reference reflector azimuth index", type=int)
parser.add_argument("cal_parameters", help="Computed polarimetric calibration parameters (in keyword:parameter format)", type=str)
args = parser.parse_args()

#Load the channels
HH, HH_par = _gpf.load_dataset(args.HH_par, args.HH)
HV, HV_par  = _gpf.load_dataset(args.HV_par, args.HV)
VH, VH_par  = _gpf.load_dataset(args.VH_par, args.VH)
VV, VV_par  = _gpf.load_dataset(args.VV_par, args.VV)

#Load the topo phase
topo_phase, topo_par  = _gpf.load_dataset(args.topo_phase_par, args.topo_phase)
cc, topo_par  = _gpf.load_dataset(args.cc, args.topo_phase)

#Construct the scattering matrix
C_matrix = _np.zeros(HH.shape + (4,4), dtype=HH.dtype)

chan_list = [[HH, HH_par], [HV, HV_par], [VH, VH_par], [VV, VV_par]]
for idx_1, (chan_1, par_1) in enumerate(chan_list):
    for idx_2, (chan_2, par_2) in enumerate(chan_list):
        #Compute pase center of channel
        ph_center_1 = _gpf.compute_phase_center(par_1)
        ph_center_2 = _gpf.compute_phase_center(par_2)
        bl = (ph_center_1 - ph_center_2)
        C_matrix[:,:, idx_1, idx_2] = chan_1 * chan_2.conj() * _np.exp(1j * topo_phase * bl / topo_par['antenna_separation'][0])

C_matrix = _mat.coherencyMatrix(C_matrix,basis='lexicographic')
T_matrix = C_matrix.lexicographic_to_pauli()

rgb, pal, c = _vf.dismph(C_matrix[:,:,0,3],k=0.7,sf=0.001)
_plt.figure()
_plt.imshow(rgb, interpolation='none')
_plt.show()

# Determine the HH-VV phase imbalance
hhvv_phase, hh_vv_par = _gpf.load_dataset(args.HHVV_flat_par, args.HHVV_flat)
HH_pwr, HH_pwr_par = _gpf.load_dataset(args.HH_pwr_par, args.HH_pwr)
VV_pwr, VV_pwr_par = _gpf.load_dataset(args.VV_pwr_par, args.VV_pwr)
reflector_imbalance_phase = _np.angle(hhvv_phase[args.ridx, args.azidx].conj())
reflector_imbalance_amplitude = (VV_pwr[args.ridx, args.azidx] / HH_pwr[args.ridx, args.azidx])**1/4

#Parameter dict
cal_dict = _od()
cal_dict['HHVV_phase_imbalance'] = [reflector_imbalance_phase, 'rad']
cal_dict['HHVV_amplitude_imbalance'] = reflector_imbalance_amplitude
cal_num = reflector_imbalance_amplitude * _np.exp(1j * reflector_imbalance_phase)
cal_dict['HHVV_amplitude_imbalance_real'] = _np.real(reflector_imbalance_amplitude)
cal_dict['HHVV_amplitude_imbalance_imaginary'] = _np.imag(reflector_imbalance_amplitude)

#Write calibrations parameters
_gpf.dict_to_par(cal_dict, args.cal_parameters)

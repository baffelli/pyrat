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
import pyrat.core.polfun as _pf
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
parser.add_argument("cal_parameters", help="Computed polarimetric calibration parameters (in keyword:parameter format)", type=str)
parser.add_argument("--ridx", help="Reference reflector range indices", type=int, nargs='*')
parser.add_argument("--azidx", help="Reference reflector azimuth indices", type=int, nargs='*')
args = parser.parse_args()
print(args.ridx)
print(args.azidx)

#Load the channels
HH, HH_par = _gpf.load_dataset(args.HH_par, args.HH)
HV, HV_par  = _gpf.load_dataset(args.HV_par, args.HV)
VH, VH_par  = _gpf.load_dataset(args.VH_par, args.VH)
VV, VV_par  = _gpf.load_dataset(args.VV_par, args.VV)

#Range vector
r_vec = HH_par['near_range_slc'][0] + _np.arange(HH.shape[0]) * HH_par['range_pixel_spacing'][0]

#Load the topo phase
topo_phase, topo_par  = _gpf.load_dataset(args.topo_phase_par, args.topo_phase)

cal_par = 2
#Construct the scattering matrix
C_matrix = _np.zeros(HH.shape + (4,4), dtype=HH.dtype)
C_matrix_flat = _np.zeros(HH.shape + (4,4), dtype=HH.dtype)
chan_list = [[HH, HH_par], [HV * cal_par, HV_par], [VH * cal_par, VH_par], [VV, VV_par]]
for idx_1, (chan_1, par_1) in enumerate(chan_list):
    for idx_2, (chan_2, par_2) in enumerate(chan_list):
        #Compute pase center of channel
        ph_center_1 = _gpf.compute_phase_center(par_1)
        ph_center_2 = _gpf.compute_phase_center(par_2)
        bl = (ph_center_1 - ph_center_2)
        C_matrix[:,:, idx_1, idx_2] = chan_1 * chan_2.conj()
        topo_phase_rescaled = _np.exp(1j * topo_phase * bl / topo_par['antenna_separation'][0])
        C_matrix_flat[:,:, idx_1, idx_2] =  C_matrix[:,:, idx_1, idx_2] * topo_phase_rescaled




C_matrix = _mat.coherencyMatrix(C_matrix,basis='lexicographic').boxcar_filter([5,10]).normalize()
C_matrix_flat = _mat.coherencyMatrix(C_matrix_flat,basis='lexicographic').boxcar_filter([5,10]).normalize()
T_matrix = C_matrix_flat.lexicographic_to_pauli()

# _plt.imshow(_np.angle(C_matrix_flat[:,:,0,1]))
# _plt.show()

ref_idx = 2
f = (C_matrix[args.ridx[ref_idx],args.ridx[ref_idx],3,3]/C_matrix[args.ridx[ref_idx],args.azidx[ref_idx],0,0])**(1/4)
VV_HH_phase_bias = _np.angle(C_matrix[args.ridx[ref_idx], args.ridx[ref_idx], 3, 0])
g = _np.mean(C_matrix[:,:,1,1,]/C_matrix[:,:,2,2,])**(1/4)
cross_pol_bias = _np.mean(C_matrix[:,:,1,2])
#Solve for phi t and phi r
phi_t = (VV_HH_phase_bias + cross_pol_bias) / 2
phi_r = (VV_HH_phase_bias - cross_pol_bias) / 2
#Distortion matrix
distortion_matrix = _np.diag([1,f*g*_np.exp(1j * phi_t),f/g*_np.exp(1j * phi_r),f**2*_np.exp(1j * (phi_t + phi_r))])
#Invert it
distortion_matrix_inv = _np.diag(1/_np.diag(distortion_matrix))


#Correct the matrix
C_matrix_c = C_matrix_flat.transform(distortion_matrix_inv,distortion_matrix_inv.T.conj())
C_matrix_c_mono = _np.zeros(HH.shape + (3,3), dtype=_np.complex64)


#Plot some ratios
_plt.figure()
_plt.imshow(10 *_np.log10(_np.abs(C_matrix_c[:,:,0,0])/_np.abs(C_matrix_c[:,:,2,2])),cmap='RdBu')
_plt.colorbar()
_plt.show()
#Extract the submatrix
for cnt_1, idx_1 in enumerate([0,1,3]):
    for cnt_2,idx_2 in enumerate([0,1,3]):
        C_matrix_c_mono[:,:,cnt_1,cnt_2] = C_matrix_c[:,:,idx_1,idx_2]

#Display nice pol signatures
C_matrix_c = _mat.coherencyMatrix(C_matrix_c_mono, basis='lexicographic')
for idx_ref in range(len(args.ridx)):
    resu = _pf.pol_signature(C_matrix_c[args.ridx[ref_idx],args.ridx[ref_idx]])
    _vf.show_signature(resu)
    _plt.show()




#Read HHVV phase at the reflectors
HHVV_phase = C_matrix[:,:,0,3]
HHVV_phase_flat = C_matrix_flat[:,:,0,3]
HHVV_phase_flat_corr = C_matrix_c[:,:,0,3]
rgb, pal, c = _vf.dismph(HHVV_phase_flat,k=0.3,sf=0.6)
_plt.figure()
_plt.subplot(2,1,1)
# _plt.imshow(rgb, interpolation='none', origin='lower')
_plt.imshow(_np.rad2deg(_np.angle(HHVV_phase_flat_corr)), cmap='rainbow')
_plt.colorbar()
ax =_plt.gca()
_plt.subplot(2,1,2, sharex=ax, sharey=ax)
# _plt.imshow(rgb, interpolation='none', origin='lower')
_plt.imshow(rgb)
_plt.colorbar()
_plt.plot(args.azidx, args.ridx,'o')
_plt.figure()
_plt.plot(r_vec[args.ridx],_np.rad2deg(_np.angle(HHVV_phase[args.ridx, args.azidx])),'bo', label='With topography')
_plt.plot(r_vec[args.ridx],_np.rad2deg(_np.angle(HHVV_phase_flat[args.ridx, args.azidx])),'ro', label='Flattened')
_plt.plot(r_vec[args.ridx],_np.rad2deg(_np.angle(HHVV_phase_flat_corr[args.ridx, args.azidx])),'go', label='Calibrated')
_plt.legend()
_plt.grid()
_plt.show()
#
#
# #Extract phase and range
# HHVV_phase_range = _np.angle(HHVV_phase[args.ridx, args.azidx])
# r = r_vec[args.ridx]
# p =_np.polyfit(r, HHVV_phase_range,2)
# HHVV_phase_range_model = _np.polynomial.polynomial.polyval(r,p[::-1])
#
# #Plot measured and modeles
# _plt.figure()
# _plt.plot(r, _np.rad2deg(HHVV_phase_range))
# _plt.plot(r, _np.rad2deg(HHVV_phase_range_model))
# _plt.show()
#
# # Determine the HH-VV phase imbalance
# hhvv_phase, hh_vv_par = _gpf.load_dataset(args.HHVV_flat_par, args.HHVV_flat)
# HH_pwr, HH_pwr_par = _gpf.load_dataset(args.HH_pwr_par, args.HH_pwr)
# VV_pwr, VV_pwr_par = _gpf.load_dataset(args.VV_pwr_par, args.VV_pwr)
# reflector_imbalance_phase = _np.angle(hhvv_phase[args.ridx, args.azidx].conj())
# reflector_imbalance_amplitude = (VV_pwr[args.ridx, args.azidx] / HH_pwr[args.ridx, args.azidx])**1/4

#Parameter dict

cal_dict = _od()
# cal_dict['HHVV_phase_imbalance'] = [reflector_imbalance_phase, 'rad']
# cal_dict['HHVV_amplitude_imbalance'] = reflector_imbalance_amplitude
# cal_num = reflector_imbalance_amplitude * _np.exp(1j * reflector_imbalance_phase)
# cal_dict['HHVV_amplitude_imbalance_real'] = _np.real(reflector_imbalance_amplitude)
# cal_dict['HHVV_amplitude_imbalance_imaginary'] = _np.imag(reflector_imbalance_amplitude)

#Write calibrations parameters
_gpf.dict_to_par(cal_dict, args.cal_parameters)

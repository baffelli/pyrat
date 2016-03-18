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

parser.add_argument("c_root", help="Flattened covariance root path", type=str)
parser.add_argument("cal_parameters", help="Computed polarimetric calibration parameters (in keyword:parameter format)",
                    type=str)
subparsers = parser.add_subparsers(help='Two modes:', dest='subparser_name')
#Subparser to measure the calibration params
subparser_measure = subparsers.add_parser('measure', help='Determine calibration parameters')
subparser_measure.add_argument("ridx", help="Reference reflector range indices", type=int, nargs=1)
subparser_measure.add_argument("azidx", help="Reference reflector azimuth indices", type=int, nargs=1)
subparser_apply = subparsers.add_parser("apply_C", help='If set, apply the calibration parameter to the input dataset to produce calibrated covariance matrices')
subparser_apply.add_argument('out_root', help='The root filename of the calibrated dataset')
#Subparser to apply them
args = parser.parse_args()
c_par = args.c_root + '.par'



if args.subparser_name == 'measure':
    C_matrix_flat = _mat.coherencyMatrix(args.c_root, c_par, basis='lexicographic', gamma=True, bistatic=True)
    print(C_matrix_flat.shape)
    av_win =[5,10]#averaging window
    C_matrix_flat_av = C_matrix_flat.boxcar_filter(av_win)
    f = (C_matrix_flat_av[args.ridx,args.azidx,3,3]/C_matrix_flat_av[args.ridx,args.azidx
    ,0,0])**(1/4.0)
    VV_HH_phase_bias = _np.angle(C_matrix_flat[args.ridx, args.azidx, 3, 0])
    g = _np.mean(C_matrix_flat_av[:,:,1,1]/C_matrix_flat_av[:,:,2,2])**(1/4.0)
    cross_pol_bias = _np.angle(_np.mean(C_matrix_flat_av[:,:,1,2]))
    #Solve for phi t and phi r
    phi_t = (VV_HH_phase_bias + cross_pol_bias) / 2
    phi_r = (VV_HH_phase_bias - cross_pol_bias) / 2
    cal_dict = _od()
    print(g)
    cal_dict['f'] = f.real
    cal_dict['g'] = g.real
    cal_dict['transmit_phase_imbalance'] = phi_t
    cal_dict['receive_phase_imbalance'] = phi_r

    #Write calibrations parameters
    _gpf.dict_to_par(cal_dict, args.cal_parameters)
elif args.subparser_name == 'apply_C' :
    C_matrix_flat = _mat.coherencyMatrix(args.c_root, c_par, basis='lexicographic', gamma=True, bistatic=True)
    cal_dic = _gpf.par_to_dict(args.cal_parameters)
    f = cal_dic['f']
    g = cal_dic['g']
    phi_t = cal_dic['transmit_phase_imbalance']
    phi_r = cal_dic['receive_phase_imbalance']
    #Distortion matrix
    distortion_matrix = _np.diag([1,f*g*_np.exp(1j * phi_t),f/g*_np.exp(1j * phi_r),f**2*_np.exp(1j * (phi_t + phi_r))])
    #Invert it
    distortion_matrix_inv = _np.diag(1/_np.diag(distortion_matrix))
    # #Correct the matrix
    C_matrix_c = C_matrix_flat.transform(distortion_matrix_inv,distortion_matrix_inv.T.conj())
    #Write the channels to a file
    C_matrix_c.to_gamma(args.out_root, bistatic=True)

    # T_matrix = C_matrix_c.boxcar_filter([5,5]).lexicographic_to_pauli()
    # T_matrix[_np.isnan(T_matrix)] = 0.00001
    # H, anisotropy, alpha_m, beta_m, p, w = T_matrix.cloude_pottier()
    # rgb = T_matrix.boxcar_filter([5,5]).pauli_image(k=0.1,sf=3)
    # #dfgfd
    # _plt.figure()
    # _plt.subplot(3,1,1)
    # _plt.imshow(rgb, interpolation='none', aspect=1.5)
    # ax = _plt.gca()
    # _plt.subplot(3,1,2, sharex=ax, sharey=ax)
    # _plt.imshow(alpha_m,cmap='rainbow',vmin=0,vmax=_np.pi/2, aspect=1.5)
    # _plt.subplot(3,1,3, sharex=ax, sharey=ax)
    # _plt.imshow(H,cmap='gray',vmin=0,vmax=1, aspect=1.5)
    # _plt.show()


# C_matrix_c_mono = _np.zeros(HH.shape + (3,3), dtype=_np.complex64)
#
# #Extract the submatrix (monostatic equivalent)
# for cnt_1, idx_1 in enumerate([0,1,3]):
#     for cnt_2,idx_2 in enumerate([0,1,3]):
#         C_matrix_c_mono[:,:,cnt_1,cnt_2] = C_matrix_c[:,:,idx_1,idx_2]
#
#
#
# #Display nice pol signatures
# C_matrix_c = _mat.coherencyMatrix(C_matrix_c_mono, basis='lexicographic')
# # import polarimetricVisualization as pV
# # pV.polarimetricVisualization(C_matrix_c, vf=_pf.pol_signature)
# # _plt.show()
# #
# for idx_ref in range(len(args.ridx)):
#     resu = _pf.pol_signature(C_matrix_c[args.ridx[ref_idx],args.azidx[ref_idx]])
#     _vf.show_signature(resu)
#     _plt.show()
#
#
# #Read HHVV phase at the reflectors
# HHVV_phase = C_matrix[:,:,0,3]
# HHVV_phase_flat = C_matrix_flat[:,:,0,3]
# HHVV_phase_flat_corr = C_matrix_c[:,:,0,3]
# rgb, pal, c = _vf.dismph(HHVV_phase_flat_corr,k=0.3)
# _plt.imshow(rgb, origin='lower')
# _plt.colorbar()
# _plt.plot(args.azidx, args.ridx,'o')
# #Vector of incidence angles
# inc_vec = _np.rad2deg(inc[args.azidx ,args.ridx])
# f, (ampl_plot, ph_plot) = _plt.subplots(2,sharex=True)
# ph_plot.plot(inc_vec,_np.rad2deg(_np.angle(HHVV_phase_flat[args.ridx, args.azidx])),'ro', label='HH-VV phase')
# ph_plot.plot(inc_vec,_np.rad2deg(_np.angle(HHVV_phase_flat_corr[args.ridx, args.azidx])),'go', label='Calibrated')
# ampl_plot.plot(inc_vec, (C_matrix[args.ridx, args.azidx,0,0]/C_matrix[args.ridx, args.azidx,3,3])**0.25,'o')
# ampl_plot.set_ylim(1,1.5)
# _plt.legend()
# ph_plot.xaxis.set_label_text(r'Incidence Angle[deg]')
# ph_plot.yaxis.set_label_text(r'HH-VV difference[deg]')
# ampl_plot.yaxis.set_label_text(r'HH-VV amplitude imbalance')
#
# _plt.show()

# T_matrix_cal = C_matrix_c.lexicographic_to_pauli()
# rgb = T_matrix_cal.pauli_image(sf=0.1)
# _plt.figure()
# _plt.imshow(rgb)
# _plt.show()
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



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
    av_win = [5,5]#averaging window
    C_matrix_flat_av = C_matrix_flat.boxcar_filter(av_win)
    f, g, phi_t, phi_r = _cal.measure_imbalance(C_matrix_flat, [args.ridx, args.azidx])
    cal_dict = _od()
    cal_dict['f'] = f.real
    cal_dict['g'] = g.real
    cal_dict['transmit_phase_imbalance'] = phi_t
    cal_dict['receive_phase_imbalance'] = phi_r

    #Write calibrations parameters
    _gpf.dict_to_par(cal_dict, args.cal_parameters)
elif args.subparser_name == 'apply_C' :
    C_matrix_flat = _mat.coherencyMatrix(args.c_root, c_par, basis='lexicographic', gamma=True, bistatic=True)
    cal_dic = _gpf.par_to_dict(args.cal_parameters)
    k = cal_dic['k'][0] +  1j * cal_dic['k'][1]
    alpha = cal_dic['alpha'][0] +  1j * cal_dic['alpha'][1]
    # phi_t = cal_dic['transmit_phase_imbalance']
    # phi_r = cal_dic['receive_phase_imbalance']
    #Distortion matrix
    distortion_matrix = _cal.distortion_matrix_for_covariance(k, alpha)
    #Invert it
    distortion_matrix_inv = _np.diag(1/_np.diag(distortion_matrix))
    # #Correct the matrix
    C_matrix_c = C_matrix_flat.transform(distortion_matrix_inv,
                                         distortion_matrix_inv.T.conj())
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






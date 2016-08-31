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






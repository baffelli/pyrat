#!/usr/bin/python
__author__ = 'baffelli'
C = 299792458.0    #speed of light m/s
KU_WIDTH = 15.798e-3 #WG-62 Ku-Band waveguide width dimension
KU_DZ = 10.682e-3   #Ku-Band Waveguide slot spacing
RANGE_OFFSET= 3

import sys, os
import numpy as _np
import argparse
import scipy as _sp
import scipy.signal as _sig
sys.path.append(os.path.expanduser('~/PhD/trunk/Code/'))
import pyrat.fileutils.gpri_files as _gpf
import pyrat.gpri_utils.calibration as _cal
from collections import namedtuple as _nt
import scipy.signal as _sig
import scipy.ndimage as _ndim
import matplotlib.pyplot as _plt
import pyrat.visualization.visfun as _vf


def first_derivative(r_arm, r_ph, r_sl):
    r_ant = _np.sqrt(r_arm**2 + r_ph**2)
    denom = r_sl
    alpha = _np.arctan(r_arm / r_ph)
    num = 2 * r_ant * r_sl * _np.cos(alpha)
    return num/denom

def second_derivative(r_arm, r_ph, r_sl):
    r_ant = _np.sqrt(r_arm**2 + r_ph**2)
    denom = r_sl**(3/2) * 4
    num = r_ant**2 * 2 * r_sl - first_derivative(r_arm, r_ph, r_sl)
    return num/denom


class gpriAzimuthProcessor:

    def __init__(self, args):
        #Load slc
        self.slc = _gpf.gammaDataset(args.slc_par, args.slc).astype(_np.complex64)
        self.args = args



    def correct(self):
        #Filtered slc
        seq = -1
        slc_filt = self.slc * 1
        #Construct range vector
        r_vec = self.slc.near_range_slc[0] + _np.arange(self.slc.shape[0]) * self.slc.range_pixel_spacing[0]
        #process each range line
        theta = _np.arange(-self.args.ws/2, self.args.ws/2) * _np.deg2rad(self.slc.GPRI_az_angle_step[0])
        for idx_r in range(self.slc.shape[0]):
            filt, dist = _cal.rep_2(self.args.r_ant, self.args.r_ph, r_vec[idx_r], theta, wrap=False)
            dist_1 = first_derivative(self.args.r_ant, self.args.r_ph, r_vec[idx_r])
            dist_2 = second_derivative(self.args.r_ant, self.args.r_ph, r_vec[idx_r])
            lam = (3e8) /17.2e9
            mf = _np.exp(4j * _np.pi * 1/lam * (dist_1 * theta + dist_2 * theta**2))
            mf_1 = _np.exp(4j * _np.pi * dist/lam)
            slc_filt[idx_r, :] = _sig.fftconvolve(self.slc[idx_r, :], mf_1, mode='same')
            if idx_r % 1000 == 0:
                    print('Processing range index: ' + str(idx_r))
            seq = seq * -1
        return slc_filt


def main():
    #Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('slc',
                help="Uncorrected slc file")
    parser.add_argument('slc_par',
                help="Slc parameters")
    parser.add_argument('slc_out', type=str,
                help="Output slc")
    parser.add_argument('slc_par_out', type=str,
                help="Output slc parameters")
    parser.add_argument('--r_ant', type=float, default=0.25,
                help="Antenna rotation arm length")
    parser.add_argument('--r_ph',
                help="Antenna phase center horizontal displacement",
                type=float, default=0.15)
    parser.add_argument('-w','--win_size', dest='ws',
                help="Convolution window size",
                type=int, default=10)
    #Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    #Create processor object
    proc = gpriAzimuthProcessor(args)
    slc_corr = proc.correct()
    with open(args.slc_out, 'wb') as of:
        slc_corr.T.astype(_gpf.type_mapping['FCOMPLEX']).tofile(of)
    _gpf.dict_to_par(proc.slc.__dict__, args.slc_par_out)
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

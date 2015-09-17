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

class gpriAzimuthProcessor:

    def __init__(self, args):
        #Load slc
        self.slc = _gpf.gammaDataset(args.slc_par, args.slc).astype(_np.complex64)
        self.args = args



    def correct(self):
        #Filtered slc
        slc_filt = self.slc * 1
        #Construct range vector
        r_vec = self.slc.near_range_slc[0] + _np.arange(self.slc.shape[0]) * self.slc.range_pixel_spacing[0]
        #process each range line
        theta = _np.arange(-self.args.ws/2, self.args.ws/2) * _np.deg2rad(self.slc.GPRI_az_angle_step[0])
        for idx_r in range(self.slc.shape[0]):
            filt, dist = _cal.rep_2(self.args.r_ant, self.args.r_ph, r_vec[idx_r], theta, wrap=False)
            slc_filt[idx_r, :] = _sig.fftconvolve(self.slc[idx_r, :], _np.exp(-1j*filt), mode='same')
            if idx_r % 1000 == 0:
                    print('Processing range index: ' + str(idx_r))
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

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

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
import scipy.optimize as _opt
import matplotlib.pyplot as plt

class gpriPlotter:

    def __init__(self, args):
        self.slc = _gpf.gammaDataset(args.slc_par, args.slc)
        self.ridx = args.ridx
        self.azidx = args.azidx
        self.ws = args.ws
        self.figpath = args.figpath

    def plot(self):
        #Slice the slc
        slc_sl = (self.ridx, slice(self.azidx - self.ws / 2, self.azidx + self.ws))
        #Determine true maximum
        max_idx = _np.argmax(_np.abs(self.slc[slc_sl]))
        #Determine half power beamwidth
        reflector_slice = self.slc[slc_sl]
        half_pwr_idx = _np.nonzero(_np.abs(reflector_slice) > _np.abs(reflector_slice[max_idx]) * 0.5)
        max_phase = _np.angle(reflector_slice[max_idx])
        #Slice slc
        reflector_slice = reflector_slice[half_pwr_idx]
        #Determine parameters
        r_vec = self.slc.near_range_slc[0] + _np.arange(self.slc.shape[0]) * self.slc.range_pixel_spacing[0]
        az_vec = _np.deg2rad(self.slc.GPRI_az_angle_step[0]) * _np.arange(-len(reflector_slice)/2 ,len(reflector_slice)/2)
        refl_ph = _np.angle(reflector_slice)
        refl_amp = (_np.abs(reflector_slice))
        f = plt.figure()
        # plt.subplot(211)
        # plt.plot(az_vec,refl_amp)
        # plt.ylabel(r'Amplitude')
        # plt.xlabel(r'azimuth angle from maximum [deg]')
        # plt.subplot(212)
        plt.plot(az_vec,_np.rad2deg(refl_ph - max_phase))
        plt.ylabel(r'Phase [deg]')
        plt.xlabel(r'azimuth angle from maximum [deg]')
        plt.ylim([-180,180])
        plt.show()
        f.savefig(self.figpath)
        plt.close(f)



def main():
    #Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('slc', type=str,
                help="SLC channel file")
    parser.add_argument('slc_par', type=str,
                help="SLC channel file parameters")
    parser.add_argument('ridx', type=float,
                help="Point target range location")
    parser.add_argument('azidx', type=float,
                help="Point target azimuth location")
    parser.add_argument('figpath', type=str,
                help="Path to save the plots")
    parser.add_argument('-w', '--win_size', dest='ws', type=float, default=20,
                help="Estimation window size")
    #Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    #Create processor object
    proc = gpriPlotter(args)
    slc_dec = proc.plot()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

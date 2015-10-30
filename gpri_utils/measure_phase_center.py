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
from matplotlib import style as _sty
_sty.use('/home/baffelli/PhD_work/Code/paper_rc.rc')


class gpriEstimator:

    def __init__(self, args):
        self.slc = _gpf.gammaDataset(args.slc_par, args.slc)
        self.u = args.u
        self.r_arm = args.r_ant
        self.ridx = args.ridx
        self.azidx = args.azidx
        self.ws = args.ws
        self.figpath = args.fig_path

    def determine(self):
        def cf(r_arm, r_ph, r, az_vec, off, meas_phase):
            sim_phase, dist = _cal.distance_from_phase_center(r_arm, r_ph, r, az_vec, wrap = False)
            cost = _np.mean(_np.abs(sim_phase  + off - meas_phase)**2)
            return cost
        #Slice the slc
        slc_sl = (self.ridx, slice(self.azidx - self.ws / 2, self.azidx + self.ws))
        #Determine true maximum
        max_idx = _np.argmax(_np.abs(self.slc[slc_sl]))
        #Determine half power beamwidth
        reflector_slice = self.slc[slc_sl]
        half_pwr_idx = _np.nonzero(_np.abs(reflector_slice) > _np.abs(reflector_slice[max_idx]) * 0.5)
        #Slice slc
        reflector_slice = reflector_slice[half_pwr_idx]
        #Determine parameters
        r_vec = self.slc.near_range_slc[0] + _np.arange(self.slc.shape[0]) * self.slc.range_pixel_spacing[0]
        az_vec = _np.deg2rad(self.slc.GPRI_az_angle_step[0]) * _np.arange(-len(reflector_slice)/2 ,len(reflector_slice)/2)
        if self.u:
            refl_ph = _np.unwrap(_np.angle(reflector_slice))
        else:
            refl_ph = _np.angle(reflector_slice)
        refl_amp = (_np.abs(reflector_slice))
        r_sl = r_vec[self.ridx]
        cost_VV = lambda par_vec: cf(self.r_arm, par_vec[0], r_vec[self.ridx], az_vec, par_vec[1], refl_ph)
        res = _opt.minimize(cost_VV, [0,0], bounds=((-2,2),(None,None)))
        print(res.x[0])
        sim_ph, dist = _cal.distance_from_phase_center(self.r_arm, res.x[0],  r_sl, az_vec, wrap=False)
        f = plt.figure()
        plt.plot(_np.rad2deg(az_vec),_np.rad2deg(refl_ph), label=r'Measured')
        plt.plot(_np.rad2deg(az_vec),_np.rad2deg(sim_ph + res.x[1]), label=r'Model')
        plt.grid()
        plt.ylabel(r'Phase [deg]')
        plt.xlabel(r'azimuth angle from maximum [deg]')
        plt.legend()
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
    parser.add_argument('r_ant', type=float,
                help="Antenna rotation arm length")
    parser.add_argument('fig_path',
                help="Path to save the graphical output", type=str)
    parser.add_argument('-w', '--win_size', dest='ws', type=float, default=20,
                help="Estimation window size")
    parser.add_argument('-u', '--unwrap', dest='u', default=False, action='store_true',
                help="Toggles phase unwrapping for the estimation")
    #Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    #Create processor object
    proc = gpriEstimator(args)
    slc_dec = proc.determine()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

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
import pyrat.fileutils.gpri_files as _gpf
import pyrat.gpri_utils.calibration as _cal
from collections import namedtuple as _nt
import scipy.optimize as _opt
import matplotlib.pyplot as plt
from matplotlib import style as _sty


#
class gpriEstimator:

    def __init__(self, args):
        self.slc = _gpf.gammaDataset(args.slc_par, args.slc)
        self.u = args.u
        self.r_arm = _gpf.xoff + _np.cos(_np.deg2rad(self.slc.GPRI_ant_elev_angle[0])) * _gpf.ant_radius
        self.ridx = args.ridx
        self.azidx = args.azidx
        self.ws = args.ws
        self.figpath = args.fig_path
        self.args = args

    def determine(self):
        def cf(r_arm, r_ph, r, az_vec, off, meas_phase):
            sim_phase, dist = _cal.distance_from_phase_center(r_arm, r_ph, r, az_vec, wrap = False)
            cost = _np.mean(_np.abs(sim_phase  + off - meas_phase)**2)
            return cost
        #
        # #Show the slc with the point highlighted
        # plt.figure()
        # plt.imshow(_np.abs(self.slc)**0.2, origin='lower')
        # plt.plot(self.azidx, self.ridx,'ro')
        # plt.show()
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
        refl_ph = _np.angle(reflector_slice)
        if self.u:
            refl_ph = _np.unwrap(refl_ph)
        else:
            refl_ph = refl_ph
        refl_ph -= refl_ph[reflector_slice.shape[0]/2]
        refl_amp = (_np.abs(reflector_slice))
        r_sl = r_vec[self.ridx]
        #Define cost function
        cost_VV = lambda par_vec: cf(self.r_arm, par_vec[0], r_vec[self.ridx], az_vec, par_vec[1], refl_ph)
        #Solve optimization problem
        res = _opt.minimize(cost_VV, [0,0], bounds=((-2,2),(None,None)))
        print(res)
        par_dict = {}
        par_dict['phase_center_offset'] = [res.x[0], 'm']
        par_dict['lever_arm_length'] = [self.r_arm, 'm']
        par_dict['range_of_closest_approach'] = [r_sl, 'm']
        _gpf.dict_to_par(par_dict, self.args.par_path)
        sim_ph, dist = _cal.distance_from_phase_center(self.r_arm, res.x[0],  r_sl, az_vec, wrap=False)
        if self.args.sf == '':
            st = '/home/baffelli/PhD/trunk/Code/paper_rc.rc'
        else:
           st =  self.args.sf
        with _sty.context(st):
            f = plt.figure()
            plt.plot(_np.rad2deg(az_vec),_np.rad2deg(refl_ph), label=r'Measured')
            plt.plot(_np.rad2deg(az_vec),_np.rad2deg(sim_ph + res.x[1]), label=r'Model')
            plt.ylabel(r'Phase [deg]')
            plt.xlabel(r'azimuth angle from maximum [deg]')
            plt.ylim(-25,25)
            plt.legend()
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
    # parser.add_argument('r_ant', type=float,
    #             help="Antenna rotation arm length")
    parser.add_argument('par_path',
                help="Path to save the phase center location", type=str)
    parser.add_argument('fig_path',
                help="Path to save the graphical output", type=str)
    parser.add_argument('-w', '--win_size', dest='ws', type=float, default=20,
                help="Estimation window size")
    parser.add_argument('-u', '--unwrap', dest='u', default=False, action='store_true',
                help="Toggles phase unwrapping for the estimation")
    parser.add_argument('--sf', help='Style file for plots', type=str, default='')
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

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
from collections import namedtuple as _nt
import scipy.optimize as _opt
import matplotlib.pyplot as _plt
import matplotlib as _mpl
from matplotlib import style as _sty
import pyrat.visualization.visfun as _vf
import pyrat.core.corefun as _cf
import pyrat.gpri_utils.calibration as cal
_sty.use('/home/baffelli/PhD/trunk/Code/paper_rc.rc')

class gpriPlotter:

    def __init__(self, args):
        self.args = args
        self.slc = _gpf.gammaDataset(args.slc_par, args.slc)
        self.ridx = args.ridx
        self.azidx = args.azidx
        self.ws = args.ws
        self.figpath = args.figpath
        self.phase_limits = args.phase_limits
        self.step_size = args.step_size

    def plot(self):
        with _sty.context('/home/baffelli/PhD/trunk/Code/paper_rc.rc'):
            f, (phase_ax, amp_ax) = _plt.subplots(2, sharex=True)
            for ridx, azidx in zip(self.args.ridx, self.args.azidx):
                #Slice the slc
                slc_sl = (ridx, slice(azidx - self.ws / 2, azidx + self.ws/2))
                subimage = self.slc[slc_sl]
                #Determine true maximum in the slice
                max_idx = _np.argmax(_np.abs(subimage))
                #Extract entire image
                #Determine the shift
                shift = subimage.shape[0] / 2 - max_idx
                slc_slc_new = slc_sl = (ridx, slice(azidx - shift - self.ws / 2, azidx - shift + self.ws/2))
                reflector_slice = self.slc[slc_slc_new]
                #Slice slc
                #Azimuth angle vector for plot
                az_vec = self.slc.GPRI_az_angle_step[0] * _np.arange(-len(reflector_slice)/2
                                                                                  ,len(reflector_slice)/2)
                refl_ph = _np.angle(reflector_slice)
                if self.args.unwrap:
                    refl_ph = _np.unwrap(refl_ph)
                else:
                    refl_ph =_np.angle(refl_ph)
                refl_ph -= refl_ph[reflector_slice.shape[0]/2]
                max_phase = refl_ph[reflector_slice.shape[0]/2]
                refl_amp = (_np.abs(reflector_slice))**2
                r_sl = self.slc.r_vec[ridx]
                line, = phase_ax.plot(az_vec,_np.rad2deg(refl_ph), label=r"r={} m".format(round(r_sl)))
                amp_ax.plot(az_vec, refl_amp/refl_amp[reflector_slice.shape[0]/2])
                ph, d = cal.distance_from_phase_center(0.22, 0.02, r_sl, _np.deg2rad(az_vec), 0)
                phase_ax.plot(az_vec, _np.rad2deg(ph - ph[ph.shape[0]/2]),color=line.get_color(),ls='--')
            #Plot line for beamwidth
            phase_ax.set_ylim(-30,30)
            phase_ax.axvline(0.2, color='red', ls='--')
            phase_ax.axvline(-0.2, color='red', ls='--')
            phase_ax.yaxis.set_label_text(r'Phase [deg]')
            phase_ax.xaxis.set_label_text(r'azimuth angle from maximum [deg]')
            amp_ax.axvline(0.2, color='red', ls='--')
            amp_ax.axvline(-0.2, color='red', ls='--')
            amp_ax.yaxis.set_label_text(r'Relative Intensity')
            amp_ax.xaxis.set_label_text(r'azimuth angle from maximum [deg]')
            phase_ax.legend()
            f.savefig(self.figpath)
            _plt.close(f)


def main():
    #Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Plot azimuth phase response at a given location')
    parser.add_argument('slc', type=str,
                help="SLC channel file")
    parser.add_argument('slc_par', type=str,
                help="SLC channel file parameters")
    parser.add_argument('--ridx', type=float, nargs='+',
                help="Point target range locations")
    parser.add_argument('--azidx', type=float,
                help="Point target azimuth locations", nargs='+')
    parser.add_argument('figpath', type=str,
                help="Path to save the plots")
    parser.add_argument('-w', '--win_size', dest='ws', type=float, default=50,
                help="Estimation window size")
    parser.add_argument('-p', '--phase_limits', dest='phase_limits', type=float, nargs=2, default=[-180, 180],
                help="Estimation window size")
    parser.add_argument('-s', '--step_size', dest='step_size', type=float, default=10,
                help="Plot y axis step size")
    parser.add_argument( '--unwrap', default=True, action='store_true',
                help="Unwrap the data when plotting")

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

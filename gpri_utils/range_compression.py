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
from collections import namedtuple as _nt

class gpriRangeProcessor:

    def __init__(self, args):
        #Load raw parameters and convert them into namedtuple
        raw_dict = _gpf.par_to_dict(args.raw_par)
        self.grp = _nt('GenericDict', raw_dict.keys())(**raw_dict)
        self.nsamp = self.grp.CHP_num_samp
        self.block_length = self.nsamp + 1
        self.chirp_duration = self.block_length/self.grp.sample_rate
        self.win =  _sig.kaiser(self.nsamp, args.kbeta)
        self.zero = args.zero
        self.win2 = _sig.hanning(2*self.zero);		#window to remove transient at start of echo due to sawtooth frequency sweep
        self.pn1 = _np.arange(self.nsamp/2 + 1) 		#list of slant range pixel numbers
        self.rps = (self.grp.ADC_sample_rate/self.nsamp*C/2.)/self.grp.RF_chirp_rate #range pixel spacing
        self.slr = (self.pn1 * self.grp.ADC_sample_rate/self.nsamp*C/2.)/self.grp.RF_chirp_rate  + RANGE_OFFSET  #slant range for each sample
        self.scale = (abs(self.slr)/self.slr[self.nsamp/8])**1.5     #cubic range weighting in power
        self.ns_min = int(round(args.r/self.rps))	#round to the nearest range sample
        self.rmin = self.ns_min * self.rps;
        self.ns_max = int(round(0.90 * self.nsamp/2))	#default maximum number of range samples for this chirp
        self.rmax = self.ns_max * self.rps;		#default maximum slant range
        self.tcycle = (self.nsamp + 1)/self.grp.ADC_sample_rate    #time/cycle
        self.dec = args.d
        self.sizeof_data = _np.dtype(_np.int16).itemsize
        self.bytes_per_record = self.sizeof_data * self.block_length  #number of bytes per echo
        #Number of lines
        self.filesize = os.path.getsize(args.corr)
        self.nl_tot = int(self.filesize/self.bytes_per_record)
        #Open raw data
        self.rawdata = _np.fromfile(args.corr, dtype = _np.int16).reshape([self.nsamp - 1, self.nl_tot])
        #Stuff for angle
        if self.grp.antenna_end != self.grp.antenna_start:
            self.ang_acc = self.grp.TSC_acc_ramp_angle
            rate_max = self.grp.TSC_rotation_speed
            t_acc = self.grp.TSC_acc_ramp_time
            self.ang_per_tcycle = self.tcycle * self.grp.TSC_rotation_speed	#angular sweep/transmit cycle
        else:
            t_acc = 0.0
            self.ang_acc = 0.0
            rate_max = 0.0
            self.ang_per_tcycle = 0.0
        if self.grp.ADC_capture_time == 0.0:
            angc = abs(self.grp.antenna_end - self.grp.antenna_start) - 2 * self.ang_acc	#angular sweep of constant velocity
            tc = abs(angc/rate_max)			#duration of constant motion
            self.grp.capture_time = 2 * t_acc + tc 	#total time for scanner to complete scan
        self.nl_acc = int(t_acc/(self.tcycle*self.dec))
        self.nl_tot_dec = int(self.grp.capture_time/(self.tcycle * self.dec))
        self.nl_image = self.nl_tot_dec - 2 * self.nl_acc
        self.image_time = (self.nl_image - 1) * (self.tcycle * self.dec)

    def compress(self):
        arr_compr = _np.zeros((self.nsamp/2 + 1, self.nl_tot_dec) ,dtype=_np.complex64)
        fshift = _np.ones(self.nsamp/ 2 +1)
        fshift[1::2] = -1
        #For each azimuth
        for idx_az in range(self.nl_acc, self.nl_image + self.nl_acc):
            #Decimated pulse
            dec_pulse = _np.zeros(self.nsamp, dtype=_np.int16)
            for idx_dec in range(self.dec):
                dec_pulse = dec_pulse + self.rawdata[:, self.dec * idx_az + idx_dec]
            if self.zero > 0:
                dec_pulse[0:self.zero] = dec_pulse[0:self.zero] * self.win2[0:self.zero]
                dec_pulse[-self.zero:] = dec_pulse[-self.zero:] * self.win2[-self.zero:]
            line_comp = _np.fft.rfft(dec_pulse * self.win) * fshift
            arr_compr[idx_az, :] = (line_comp[self.ns_min:self.ns_max].conj() * self.scale[self.ns_min:self.ns_max + 1]).astype('complex64')
        return arr_compr



def main():
    #Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('corr',
                help="Squint corrected file")
    parser.add_argument('raw_par',
                help="GPRI raw file parameters")
    parser.add_argument('slc_out', type=str,
                help="Output slc")
    parser.add_argument('slc_par_out', type=str,
                help="Slc parameter file")
    parser.add_argument('-d', type=int, default=5,
                help="Decimation factor")
    parser.add_argument('-z',
                help="Number of samples to zero at the beginning of each pulse",dest='zero',
                type=int, default=300)
    parser.add_argument("-k", type=float, default=3.00, dest='kbeta',
                      help="Kaiser Window beta parameter")
    parser.add_argument('-a', help="The dataset has been computed in the ati mode, with no\
                                   azimuth interpolation")
    parser.add_argument('-r', help='Near range for the slc',type=float, default=0)
    #Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    #Create processor object
    proc = gpriRangeProcessor(args)
    slc = proc.compress()
    #Create default slc parameters
    slc_dict = _gpf.default_slc_dict()
    #Compute parameters

    with open(args.slc_out, 'wb') as of:
        slc.T.astype(_gpf.type_mapping['FCOMPLEX']).tofile(of)
    # with open(args.raw_par, 'rt') as ip, open(args.raw_par_out, 'wt') as op:
    #     op.write(ip)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

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
        #Pattern parameters
        #Load raw parameters and convert them into namedtuple
        raw_dict = _gpf.par_to_dict(args.raw_par)
        self.raw_par = _gpf.rawParameters(raw_dict, args.raw)
        self.raw_par.compute_slc_parameters(args)
        self.rawdata = _np.fromfile(args.raw, dtype=self.raw_par.dt).reshape([self.raw_par.block_length, self.raw_par.nl_tot][::-1]).T


    def compress(self):
        arr_compr = _np.zeros((self.raw_par.ns_max - self.raw_par.ns_min + 1, self.raw_par.nl_tot_dec) ,dtype=_np.complex64)
        print(arr_compr.shape)
        fshift = _np.ones(self.raw_par.nsamp/ 2 +1)
        fshift[1::2] = -1
        #For each azimuth
        for idx_az in range(self.raw_par.nl_tot_dec):
            #Decimated pulse
            dec_pulse = _np.zeros(self.raw_par.block_length, dtype=_np.int16)
            for idx_dec in range(self.raw_par.dec):
                current_idx = idx_az * self.raw_par.dec + idx_dec
                if current_idx % 1000 == 0:
                    print('Accessing azimuth index: ' + str(current_idx))
                try:
                    dec_pulse = dec_pulse + self.rawdata[:, current_idx ]
                except IndexError as err:
                    print(err.message)
                    break
            if self.raw_par.zero > 0:
                dec_pulse[0:self.raw_par.zero] = dec_pulse[0:self.raw_par.zero] * self.raw_par.win2[0:self.raw_par.zero]
                dec_pulse[-self.raw_par.zero:] = dec_pulse[-self.raw_par.zero:] * self.raw_par.win2[-self.raw_par.zero:]
            line_comp = _np.fft.rfft(dec_pulse[1::]/self.raw_par.dec * self.raw_par.win) * fshift
            arr_compr[:, idx_az] = (line_comp[self.raw_par.ns_min:self.raw_par.ns_max + 1].conj() * self.raw_par.scale[self.raw_par.ns_min:self.raw_par.ns_max + 1]).astype('complex64')
        return arr_compr

    def fill_dict(self):
        slc_dict = _gpf.default_slc_dict()
        slc_dict['range_samples'] = self.raw_par.ns_out




def main():
    #Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('raw',
                help="Raw channel file")
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
    #
    print(proc.raw_par.ns_out)
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

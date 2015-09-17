#!/usr/bin/python
__author__ = 'baffelli'
C = 299792458.0    #speed of light m/s
KU_WIDTH = 15.798e-3 #WG-62 Ku-Band waveguide width dimension
KU_DZ = 10.682e-3   #Ku-Band Waveguide slot spacing
RANGE_OFFSET= 3
ADCSR=6250000.0
RF_CHIRP_RATE=49942173063.8


import sys, os
import numpy as _np
import argparse
import scipy as _sp
import scipy.signal as _sig
sys.path.append(os.path.expanduser('~/PhD/trunk/Code/'))
import pyrat.fileutils.gpri_files as _gpf
from collections import namedtuple as _nt

class gpriBackwardsProcessor:

    def __init__(self, args):
        #Load slc
        self.slc = _gpf.gammaDataset(args.slc_par, args.slc).astype(_np.complex64)
        self.args = args
        self.range_samples = self.args.t * ADCSR
        rps = (ADCSR/self.range_samples*C/2.)/RF_CHIRP_RATE
        self.ns_min = int(round(args.rmin/rps)) * 2 + 1
        self.ns_max = int(round(args.rmax/rps)) *  2 + 1


    def decompress(self):
        arr_raw = _np.zeros((self.range_samples, self.slc.azimuth_lines), dtype=_np.int16)
        arr_transf = _np.zeros((self.slc.shape[0] * 2 + 1, self.slc.azimuth_lines))
        #First convert back
        for idx_az in range(self.slc.shape[1]):
            arr_raw[self.ns_min:self.ns_max, idx_az] = _np.fft.irfft(self.slc[:, idx_az] * 32768)
        return arr_raw



def main():
    #Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('slc',
                help="Slc file to process")
    parser.add_argument('slc_par',
                help="GPRI slc file parameters")
    parser.add_argument('raw_out', type=str,
                help="Output raw")
    parser.add_argument('raw_par_out', type=str,
                help="raw parameter file")
    parser.add_argument('-d', type=int, default=5,
                help="Decimation factor")
    parser.add_argument('-z',
                help="Number of samples to zero at the beginning of each pulse",dest='zero',
                type=int, default=300)
    parser.add_argument("-k", type=float, default=3.00, dest='kbeta',
                      help="Kaiser Window beta parameter")
    parser.add_argument("-s","--apply_scale", type=bool, default=True, dest='apply_scale')
    parser.add_argument('-r', help='Near range for the slc',dest='rmin', type=float, default=0)
    parser.add_argument('-R', help='Far range for the slc', dest='rmax',type=float, default=1000)
    parser.add_argument('-t', help='Chirp duration of the original data [s]', type=float, default=4e-3)
    #Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    #Create processor object
    proc = gpriBackwardsProcessor(args)
    raw = proc.decompress()
    with open(args.raw_out, 'wb') as of:
        raw.T.astype(_gpf.type_mapping['SHORT INTEGER']).tofile(of)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

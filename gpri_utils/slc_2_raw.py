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
        # self.slc = _gpf.gammaDataset(args.slc_par, args.slc).astype(_np.complex64)
        self.slc_par = _gpf.par_to_dict(args.slc_par)
        self.args = args
        #Compute tcycle (supposing the decimation factor is known)
        self.tcycle = 1/(self.slc_par['prf'][0])/self.args.dec
        #Supposing a fixed ADC rate, the block length and the number of samples are easiyl computed
        self.block_length = int(self.tcycle * ADCSR) + 1
        self.nsamp = self.block_length - 1
        #We have to compute the chirp rate
        self.chirp_rate = self.slc_par['chirp_bandwidth'] / self.tcycle
        rps = (ADCSR/self.nsamp*C/2.)/self.chirp_rate
        self.ns_min = int(round(args.rmin/rps)) * 2 + 1
        self.ns_max = int(round(args.rmax/rps)) *  2 + 1
        self.dt = _gpf.type_mapping[self.slc_par['image_format']]
        #Scale factor
        self.rps = (ADCSR/self.nsamp*C/2.)/self.chirp_rate #range pixel spacing
        self.pn1 = _np.arange(self.nsamp/2 + 1) 		#list of slant range pixel numbers
        self.slr = (self.pn1 * ADCSR/self.nsamp*C/2.)/self.chirp_rate  + RANGE_OFFSET  #slant range for each sample
        self.scale = (abs(self.slr)/self.slr[self.nsamp/8])**1.5     #cubic range weighting in power
        self.ns_min = int(round(args.rmin/self.rps))	#round to the nearest range sample
        if(args.rmax != 0.0):	#check if greater than maximum value for the selected chirp
          if (int(round(args.rmax/self.rps)) <= self.ns_max):
            self.ns_max = int(round(args.rmax/self.rps))
        else:
            self.ns_max = int(round(0.90 * self.nsamp/2))


    def decompress(self, of):
        arr_raw = _np.zeros(self.block_length, dtype=_np.int16)
        print(arr_raw.shape)
       # arr_transf = _np.zeros((self.slc.shape[0] * 2 + 1, self.slc.azimuth_lines))
        #First convert back
        with open(self.args.slc, 'rb') as inf:
            for idx_az in range(int(self.slc_par['azimuth_lines'])):
                line = _np.fromfile(inf,dtype=self.dt, count=int(self.slc_par['range_samples']))
                if idx_az % 1000 == 0:
                    out_str = "processing line {}".format(idx_az)
                    print(out_str)
                    print(line)
                arr_raw[1::] = _np.fft.irfft(line * self.scale[self.ns_min:self.ns_max + 1][::-1], n=self.nsamp)
                arr_raw.astype(_gpf.type_mapping['SHORT INTEGER']).tofile(of)
        #return arr_raw



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
    parser.add_argument('-d', type=int, default=1, dest='dec',
                help="Decimation factor")
    parser.add_argument('-z',
                help="Number of samples to zero at the beginning of each pulse",dest='zero',
                type=int, default=300)
    parser.add_argument("-k", type=float, default=3.00, dest='kbeta',
                      help="Kaiser Window beta parameter")
    parser.add_argument("-s","--apply_scale", type=bool, default=True, dest='apply_scale')
    parser.add_argument('-r', help='Near range for the slc',dest='rmin', type=float, default=0)
    parser.add_argument('-R', help='Far range for the slc', dest='rmax',type=float, default=1000)
    #Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    #Create processor object
    proc = gpriBackwardsProcessor(args)
    with open(args.raw_out, 'wb') as of:
        proc.decompress(of)
        #raw.T.astype(_gpf.type_mapping['SHORT INTEGER']).tofile(of)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

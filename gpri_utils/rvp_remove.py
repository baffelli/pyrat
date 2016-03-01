#!/usr/bin/python
__author__ = 'baffelli'
C = 299792458.0    #speed of light m/s
KU_WIDTH = 15.798e-3 #WG-62 Ku-Band waveguide width dimension
KU_DZ = 10.682e-3   #Ku-Band Waveguide slot spacing VV

RANGE_OFFSET= 3

import sys, os
import numpy as _np
import argparse
import scipy as _sp
import scipy.signal as _sig
import pyrat.fileutils.gpri_files as _gpf
from collections import namedtuple as _nt
#
class gpriRVPProcessor:

    def __init__(self, args):
        #Pattern parameters
        #Load raw parameters and convert them into namedtuple
        self.rawdata, raw_par = _gpf.load_raw(args.raw_par, args.raw, nchan=1)
        print(self.rawdata.shape)
        self.raw_par = _gpf.rawParameters(raw_par, args.raw)
        self.fshift = _np.ones(self.raw_par.nsamp/ 2 +1)
        self.fshift[1::2] = -1
        self.rvp = _np.exp(1j*4.*_np.pi*self.raw_par.grp.RF_chirp_rate*(self.raw_par.slr/C)**2) #residual video phase correction


    def correct(self):
        #For each azimuth
        self.rawdata_corr = 1 * self.rawdata
        for idx_az in range(0,self.raw_par.nl_tot):
            temp = _np.append(_np.fft.ifftshift(_np.fft.irfft(_np.fft.rfft((self.rawdata[1:, idx_az].astype(_np.float32))/ 32768)* self.fshift*
                                            self.rvp.astype('complex64').conj() * 32768),0) ,0)
            self.rawdata_corr[:, idx_az] = temp.astype(_gpf.type_mapping['SHORT INTEGER'])
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(2,1,1)
        plt.imshow((self.rawdata[::5,::5]))
        ax = plt.gca()
        plt.subplot(2,1,2, sharex= ax, sharey=ax)
        plt.imshow((self.rawdata_corr[::5,::5]))
        plt.show()
        return self.rawdata_corr


def main():
    #Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('raw',
                help="Raw channel file")
    parser.add_argument('raw_par',
                help="GPRI raw file parameters")
    parser.add_argument('raw_out',
                help="Raw channel file with RVP removed")
    parser.add_argument('raw_par_out',
                help="GPRI raw file parameters")
    #Read adrguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    #Create processor object
    proc = gpriRVPProcessor(args)
    raw_corr = proc.correct()
    with open(args.raw_out, 'wb') as of:
        raw_corr.astype(_gpf.type_mapping['SHORT INTEGER']).T.tofile(of)
    _gpf.dict_to_par(_gpf.par_to_dict(args.raw_par), args.raw_par_out)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

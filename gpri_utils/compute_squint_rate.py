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
import matplotlib.pyplot as _plt
class rawTracker:

    def __init__(self, args):
        self.args = args
        #Load raw file parameters
        raw_par = _gpf.par_to_dict(args.raw_par)
        self.grp = _gpf.rawParameters(raw_par, self.args.raw)
        self.raw_data = []
        self.fshift = _np.ones(self.grp.nsamp/ 2 +1)
        self.fshift[1::2] = -1
#TODO utility function to load block from binary
    def open_raw(self):
        #Seek until the window of interest
        seeksize = self.grp.bytes_per_record * (self.args.aztarg - self.args.search_window/2)
        count = int(self.grp.block_length * self.args.search_window)
        with open(self.args.raw,"rb") as f:
            f.seek(seeksize, os.SEEK_SET)
            self.raw_data = _np.fromfile(f, dtype=self.grp.dt,
                                         count=count).reshape([self.grp.block_length,
                                                               self.args.search_window][::-1]).T



    def track_maximum(self):
        self.open_raw()
        #First we compute the range spectrum
        range_spectrum = _np.fft.rfft(self.raw_data,axis=0) * self.fshift[:,None] * self.grp.scale[:,None]
        #From the range spectrum we can obtain the maximum
        mean_spectrum = _np.mean(_np.abs(range_spectrum),axis=1)
        #From the maximum energy we determine the dominant target frequency
        max_idx = _np.argmax(mean_spectrum[1:])
        freq_vec = _np.fft.rfftfreq(mean_spectrum.shape[0])
        ws = 100
        freq_filter = _np.zeros(mean_spectrum.shape)
        freq_filter[max_idx-ws/2:max_idx+ws/2] = _np.hamming(ws)
        #Filter spectrum to extract only range of interest
        filt_data = _np.fft.irfft(range_spectrum * freq_filter[:,None] * -self.fshift[:,None],axis=0)
        #COmpute its envelope
        filt_env = _sig.hilbert(filt_data,axis=0)
        #Track the maximum
        squint_vec = _np.array( [_np.argmax(_np.abs(filt_env[idx,:]))
                                 for idx in range(filt_env.shape[0])])
        #Subtract the squint at the center
        squint_vec = squint_vec - squint_vec[squint_vec.shape[0]/2]
        #Convert indices in degrees
        squint_vec = squint_vec * self.grp.ang_per_tcycle
        #Fit a polynomial
        pars = _sp.polyfit(self.grp.freq_vec,squint_vec,1)
        print(pars)
        #Computed fitted data
        _plt.figure()
        _plt.imshow(_np.abs(filt_data))
        _plt.show()
        #fitted_squint = pars[0] * self.grp.freq_vec + pars[1]
        return _np.transpose([self.grp.freq_vec,squint_vec]), pars



def main():
    #Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('raw',
                help="Raw file to process")
    parser.add_argument('raw_par',
                help="Raw file parameters to process")
    parser.add_argument('aztarg', type=int,
                help="Target of interest azimuth location")
    parser.add_argument('squint', type=str,
                help="Output squint profile (text)")
    parser.add_argument('squint_fit', type=str,
                help="Linear squint parameters (text)")
    parser.add_argument('--search-window', dest='search_window', default=200, type=int, )
    #Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    #Create processor object
    proc = rawTracker(args)
    squint_data, squint_par = proc.track_maximum()
    with open(args.squint, 'w') as of, open(args.squint_fit, 'w') as of1:
        _np.savetxt(of,squint_data,header="Frequency\tSquint")
        _np.savetxt(of1,squint_par)
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

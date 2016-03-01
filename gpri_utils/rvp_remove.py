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
        #Fast (chirp) time
        self.fast_time = _np.arange(self.raw_par.nsamp/2) * 2/self.raw_par.grp.ADC_sample_rate
        #Chirsp duration
        chirp_duration = 1/self.raw_par.grp.ADC_sample_rate * 2 * self.raw_par.nsamp/2
        #Slow time
        self.slow_time = chirp_duration * _np.arange(0,self.raw_par.nl_tot)
        #Range frequency
        self.range_freq = self.fast_time * self.raw_par.grp.RF_chirp_rate

    def correct(self):
        #For each azimuth
        self.rawdata_corr = 1 * self.rawdata
        for idx_az in range(0,self.raw_par.nl_tot):
            temp = _np.append(_np.fft.ifftshift(_np.fft.irfft(_np.fft.rfft((self.rawdata[1:, idx_az].astype(_np.float32))/ 32768)* self.fshift*
                                            self.rvp.astype('complex64').conj() * 32768),0) ,0)
            self.rawdata_corr[:, idx_az] = temp.astype(_gpf.type_mapping['SHORT INTEGER'])
        #Convert to complex
        self.rawdata_rmc = 1 * self.rawdata.astype(_np.complex64)/  32768
        #Range compressed data
        self.rc = _np.zeros((self.fshift.shape[0] , self.raw_par.nl_tot), dtype=_np.complex64)
        #Compute rcmc
        #Antenna length
        r_ant = _np.sqrt(0.25** + 0.15**2)
        #rotation speed
        omega = _np.deg2rad(self.raw_par.grp.TSC_rotation_speed)
        #fmcw rate
        K = self.raw_par.grp.RF_chirp_rate
        #Wavelength
        lam = self.raw_par.grp.RF_center_freq / C
        #Reference slow time (taken at the center of the scene)
        t_slow_ref = self.slow_time[4600]
        #Reference fast time (taken at the center of the chirp)
        t_fast_ref = self.fast_time[self.fast_time.shape[0] * 0.2]
        #First rmc term
        first_term = 2 * r_ant/C * _np.cos((self.slow_time - t_slow_ref + t_fast_ref) * omega)
        second_term = 2 * r_ant * omega / (lam * K) * _np.sin(omega * (self.slow_time - t_slow_ref))
        rmc = _np.zeros(self.rawdata.shape, dtype=_np.complex64)
        for idx_f in range(self.rawdata.shape[0] - 1):
            rmc[idx_f,:] = _np.exp(2j * _np.pi * (first_term + second_term) * self.range_freq[idx_f])
            self.rawdata_rmc[idx_f,:] = self.rawdata_rmc[idx_f,:] * rmc[idx_f, :]

        for idx_az in range(self.rawdata_rmc.shape[1]):
            self.rc[:, idx_az] = _np.fft.fft(self.rawdata_rmc[1:, idx_az], axis=0)[0:self.raw_par.nsamp/2 + 1] * self.fshift * self.raw_par.scale
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(_np.abs(self.rc[0:1200,:])**0.2)
        plt.show()
        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.imshow((self.rawdata[::5,::5]),extent=[self.slow_time.min(),self.slow_time.max(),self.fast_time.min(),self.fast_time.max()], aspect=1000, origin='upper')
        # plt.ylabel('Fast time [s]')
        # plt.xlabel('Slow time [s]')
        # plt.show()
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

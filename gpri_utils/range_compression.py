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
        self.apply_scale = args.apply_scale

    def compress(self):
        arr_compr = _np.zeros((self.raw_par.ns_max - self.raw_par.ns_min + 1, self.raw_par.nl_tot_dec) ,dtype=_np.complex64)
        print(arr_compr.shape)
        fshift = _np.ones(self.raw_par.nsamp/ 2 +1)
        fshift[1::2] = -1
        #For each azimuth
        for idx_az in range(self.raw_par.nl_tot_dec):
            #Decimated pulse
            dec_pulse = _np.zeros(self.raw_par.block_length, dtype=_np.float32)
            for idx_dec in range(self.raw_par.dec):
                current_idx = idx_az * self.raw_par.dec + idx_dec
                current_idx_1 = idx_az + idx_dec * self.raw_par.dec
                if self.raw_par.dt is _np.dtype(_np.int16):
                    current_data = self.rawdata[:, current_idx ] / 32768
                else:
                     current_data = self.rawdata[:, current_idx ]
                if current_idx % 1000 == 0:
                    print('Accessing azimuth index: {} '.format(current_idx))
                try:
                    dec_pulse = dec_pulse + current_data
                except IndexError as err:
                    print(err.message)
                    break
            if self.raw_par.zero > 0:
                dec_pulse[0:self.raw_par.zero] = dec_pulse[0:self.raw_par.zero] * self.raw_par.win2[0:self.raw_par.zero]
                dec_pulse[-self.raw_par.zero:] = dec_pulse[-self.raw_par.zero:] * self.raw_par.win2[-self.raw_par.zero:]
            #Emilinate the first sample, as it is used to jump back to the start freq
            line_comp = _np.fft.rfft(dec_pulse[1::]/self.raw_par.dec * self.raw_par.win) * fshift
            #Decide if applying the range scale factor or not
            if self.apply_scale:
                scale_factor = self.raw_par.scale[self.raw_par.ns_min:self.raw_par.ns_max + 1]
            else:
                scale_factor = 1
            arr_compr[:, idx_az] = (line_comp[self.raw_par.ns_min:self.raw_par.ns_max + 1].conj() * scale_factor).astype('complex64')
        #Remove lines used for rotational acceleration
        arr_compr = arr_compr[:, self.raw_par.nl_acc:self.raw_par.nl_image + self.raw_par.nl_acc:]
        print(arr_compr.shape)
        return arr_compr

    def fill_dict(self):
        image_time = (self.raw_par.nl_image - 1) * (self.raw_par.tcycle * self.raw_par.dec)
        slc_dict = _gpf.default_slc_dict()
        ts = self.raw_par.grp.time_start
        ymd = (ts.split()[0]).split('-')
        hms_tz = (ts.split()[1]).split('+')    #split into HMS and time zone information
        hms = (hms_tz[0]).split(':')        #split HMS string using :
        sod = int(hms[0])*3600 + int(hms[1])*60 + float(hms[2])    #raw data starting time, seconds of day
        st0 = sod + self.raw_par.nl_acc * self.raw_par.tcycle * self.raw_par.dec + \
              (self.raw_par.dec/2.0)*self.raw_par.tcycle    #include time to center of decimation window
        az_step = self.raw_par.ang_per_tcycle * self.raw_par.dec
        prf = abs(1.0/(self.raw_par.tcycle*self.raw_par.dec))
        seq = self.raw_par.grp.TX_RX_SEQ
        fadc =  C/(2.*self.raw_par.rps)
        slc_dict['title'] = ts
        slc_dict['date'] = [ymd[0], ymd[1], ymd[2]]
        slc_dict['start_time'] = [st0, 's']
        slc_dict['center_time'] = [st0 + image_time / 2 , 's']
        slc_dict['end_time'] = [st0 + image_time , 's']
        slc_dict['range_samples'] = self.raw_par.ns_out
        slc_dict['azimuth_lines'] = self.raw_par.nl_tot_dec - 2 * self.raw_par.nl_acc
        slc_dict['range_pixel_spacing'] = [self.raw_par.rps, 'm']
        slc_dict['azimuth_line_time'] = [self.raw_par.tcycle * self.raw_par.dec, 's']
        slc_dict['near_range_slc'] = [self.raw_par.rmin, 'm']
        slc_dict['center_range_slc'] = [(self.raw_par.rmin + self.raw_par.rmax)/2, 'm']
        slc_dict['far_range_slc'] = [self.raw_par.rmax, 'm']
        slc_dict['radar_frequency'] = [self.raw_par.grp.RF_center_freq, 'Hz']
        slc_dict['adc_sampling_rate'] = [fadc, 'Hz']
        slc_dict['prf'] = [prf, 'Hz']
        slc_dict['chirp_bandwidth'] = self.raw_par.grp.RF_freq_max - self.raw_par.grp.RF_freq_min
        slc_dict['receiver_gain'] = [60 - self.raw_par.grp.IMA_atten_dB, 'dB']
        slc_dict['GPRI_TX_mode'] = self.raw_par.grp.TX_mode
        slc_dict['GPRI_TX_antenna'] = seq[0]
        slc_dict['GPRI_RX_antenna'] = seq[1] + seq[3]
        slc_dict['GPRI_TX_antenna_position'] = [self.raw_par.grp.GPRI_TX_antenna_position, 'm']
        slc_dict['GPRI_RX_antenna_position'] = [self.raw_par.grp.GPRI_RX_antenna_position, 'm']
        slc_dict['GPRI_az_start_angle'] = [self.raw_par.az_start, 'degrees']
        slc_dict['GPRI_az_angle_step'] = [az_step, 'degrees']
        slc_dict['GPRI_ant_elev_angle'] = [self.raw_par.grp.antenna_elevation, 'degrees']
        slc_dict['GPRI_ref_north'] = [self.raw_par.grp.geographic_coordinates[0], 'degrees']
        slc_dict['GPRI_ref_east'] = [self.raw_par.grp.geographic_coordinates[1], 'degrees']
        slc_dict['GPRI_ref_alt'] = [self.raw_par.grp.geographic_coordinates[2], 'm']
        slc_dict['GPRI_geoid'] = [self.raw_par.grp.geographic_coordinates[3], 'm']
        return slc_dict



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
    proc = gpriRangeProcessor(args)
    slc = proc.compress()
    #
    print(proc.raw_par.ns_out)
    #Create default slc parameters
    slc_dict = proc.fill_dict()
    #Compute parameters
    print(len(slc))
    with open(args.slc_out, 'wb') as of:
        slc.T.astype(_gpf.type_mapping['FCOMPLEX']).tofile(of)
    _gpf.dict_to_par(slc_dict, args.slc_par_out)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
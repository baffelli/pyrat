#!/usr/bin/python
__author__ = 'baffelli'
RANGE_OFFSET= 3

import sys, os
import shutil as _shutil
import numpy as _np
import argparse
sys.path.append(os.path.expanduser('~/PhD/trunk/Code/'))
import pyrat.fileutils.gpri_files as _gpf



class squintCorrector():

    def __init__(self, args, grp):
        self.raw_par = grp
        self.args = args

    def correct_squint(self, raw_channel, squint_function):
        #We require a function to compute the squint angle
        squint_vec = squint_function(self.raw_par.freq_vec)
        squint_vec = squint_vec - squint_vec[self.raw_par.freq_vec.shape[0]/2]
        #Normal angle vector
        angle_vec = self.raw_par.grp.STP_antenna_start +\
                    _np.arange(raw_channel.shape[1]) * self.raw_par.ang_per_tcycle
        print(raw_channel.shape)
        #We do not correct the squint of the first sample
        squint_vec = _np.insert(squint_vec, 0,0)
        #Interpolated raw channel
        raw_channel_interp = _np.zeros_like(raw_channel)
        for idx in range(0, raw_channel.shape[0]):
            print("interp range:" + str(idx))
            az_new = angle_vec + squint_vec[idx]
            raw_channel_interp[idx,: ] = _np.interp(az_new, angle_vec, raw_channel[idx,:],left=0.0, right=0.0)
        return raw_channel_interp

#These function take a vector of frequencies and
#return a vector of squint angles in degrees
def model_squint(freq_vec):
    sq_ang = _gpf.squint_angle(freq_vec, _gpf.KU_WIDTH, _gpf.KU_DZ)

def linear_squint(freq_vec, sq_rate, center_squint):
    return sq_rate * freq_vec + center_squint

def main():
    #Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('raw',
                help="GPRI raw file")
    parser.add_argument('raw_par',
                help="GPRI raw file parameters")
    parser.add_argument('raw_out', type=str,
                help="Corrected GPRI raw file")
    parser.add_argument('raw_par_out', type=str,
                help="Parameters of the corrected GPRI raw file")
    parser.add_argument('sq_rate', nargs=1, help='Squint rate in deg/Hz', type=float)
    parser.add_argument('center_squint', nargs=1, help='Squint at the center frequency in degrees', type=float)
    #Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    #Read raw dataqset and parameters
    raw_dict = _gpf.par_to_dict(args.raw_par)
    #Compute parameters
    raw_par = _gpf.rawParameters(raw_dict, args.raw)
    raw_data = _np.fromfile(args.raw, dtype=raw_par.dt).reshape([raw_par.nl_tot,
                                                                raw_par.block_length]).T
    #Create squint corrector object
    squint_processor = squintCorrector(args, raw_par)
    #Squint correction function
    sf = lambda freq_vec: linear_squint(freq_vec, args.sq_rate, args.center_squint)
    #Call processor
    raw_data_corr = squint_processor.correct_squint(raw_data, sf)

    #Write dataset
    with open(args.raw_out, 'wb') as of:
        raw_data_corr.T.astype(_gpf.type_mapping['SHORT INTEGER']).tofile(of)
    #Copy parameters
    _shutil.copyfile(args.raw_par, args.raw_par_out)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

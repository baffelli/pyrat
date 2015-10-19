#!/usr/bin/python
__author__ = 'baffelli'
RANGE_OFFSET= 3

import sys, os
import numpy as _np
import argparse
sys.path.append(os.path.expanduser('~/PhD/trunk/Code/'))
import pyrat.fileutils.gpri_files as _gpf





def correct_squint(arr, squint_vec, angle_vec):
    """
    This function corrects the frequency dependent squint
    in the GPRI data
    """
    #Add a no squint correction to the first sample
    assert len(squint_vec) == arr.shape[0] - 1
    squint_vec = _np.insert(squint_vec, 0,0)
    arr_int = _np.zeros_like(arr)
    for idx in range(0, arr.shape[0]):
        if idx % 100 == 0:
            print("interp range:" + str(idx))
        az_new = angle_vec + squint_vec[idx]
        arr_int[idx,: ] = _np.interp(az_new, angle_vec, arr[idx,:],left=0.0, right=0.0)
    return arr_int



def correct_channel(arr, freq_vec, az_spacing):
    ang_vec = _np.arange(arr.shape[1])
    sq_ang = _gpf.squint_angle(freq_vec, _gpf.KU_WIDTH, _gpf.KU_DZ)
    sq_vec = (sq_ang - sq_ang[sq_ang.shape[0]/2]) / az_spacing
    arr_corr = correct_squint(arr, sq_vec, ang_vec)
    return arr_corr

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
    #Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    #Read raw dataqset
    raw_dict = _gpf.par_to_dict(args.raw_par)
    #Compute parameters
    raw_par = _gpf.rawParameters(raw_dict, args.raw)
    #Compute shape
    shape = [raw_par.block_length, raw_par.nl_tot]
    chan = _np.fromfile(args.raw, dtype=raw_par.dt).reshape(shape[::-1]).T
    #Azimuth spacing
    azspacing = raw_par.ang_per_tcycle
    #Empty dataset
    chan_corr = _np.zeros_like(chan) + chan
    #Apply interpolation
    chan_corr = correct_channel(chan, raw_par.freq_vec, azspacing)
    #Write dataset
    with open(args.raw_out, 'wb') as of:
        chan_corr.T.astype(_gpf.type_mapping['SHORT INTEGER']).tofile(of)
    _gpf.dict_to_par(raw_dict, args.raw_par_out)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

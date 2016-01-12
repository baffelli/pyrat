#!/usr/bin/python
__author__ = 'baffelli'


import sys, os
import numpy as _np
import argparse
import pyrat.fileutils.gpri_files as _gpf
import scipy.signal as _sig




def hilbert(arr):
    arr_h = _np.zeros_like(arr, dtype=_np.complex64)
    for idx_az in range(arr.shape[1]):
        if idx_az % 100 == 0:
            print("interp azimtuh:" + str(idx_az))
        arr_h[:, idx_az] = _sig.hilbert(arr[:, idx_az].astype(_np.float32))
    return arr_h

def main():
    #Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                help="Input file")
    parser.add_argument('width', type=int, help='complex samples per line')
    parser.add_argument('output',
                help="Output file with hilbert transformation")
    parser.add_argument('dt',
                help="Data type 0 int16 \n 1 float 32", default=0)
    #Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    if args.dt is 0:
        data_type = _gpf.type_mapping['SHORT INTEGER']
    else:
        data_type = _gpf.type_mapping['FLOAT']
    with open(args.output, 'wb') as of, open(args.input, 'rb') as inf:
        chan = _np.fromfile(inf, dtype=data_type).astype(_np.int16)
        chan_nsamp = len(chan) / args.width
        chan = chan.reshape([args.width, chan_nsamp])
        chan_h = hilbert(chan)
        chan_h = chan_h.astype(_gpf.type_mapping["SCOMPLEX"])
        chan_h.tofile(of)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

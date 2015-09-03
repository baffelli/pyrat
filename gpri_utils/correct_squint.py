#!/usr/bin/python
__author__ = 'baffelli'
C = 299792458.0    #speed of light m/s
KU_WIDTH = 15.798e-3 #WG-62 Ku-Band waveguide width dimension
KU_DZ = 10.682e-3   #Ku-Band Waveguide slot spacing
RANGE_OFFSET= 3

import sys, os
import numpy as _np
import argparse
sys.path.append(os.path.expanduser('~/PhD/trunk/Code/'))
import pyrat.fileutils.gpri_files as _gpf



def lamg(freq, w):
    """
    This function computes the wavelength in waveguide for the TE10 mode
    """
    la = lam(freq)
    return la / _np.sqrt(1.0 - (la / (2 * w))**2)	#wavelength in WG-62 waveguide

#lambda in freespace
def lam(freq):
    """
    This function computes the wavelength in freespace
    """
    return C/freq

def squint_angle(freq, w, s):
    """
    This function computes the direction of the main lobe of a slotted
    waveguide antenna as a function of the frequency, the size and the slot spacing.
    It supposes a waveguide for the TE10 mode
    """
    sq_ang = _np.arccos(lam(freq) / lamg(freq, w) -  lam(freq) / (2 * s))
    dphi = _np.pi *(2.*s/lamg(freq, w) - 1.0)				#antenna phase taper to generate squint
    sq_ang_1 = _np.rad2deg(_np.arcsin(lam(freq) *dphi /(2.*_np.pi*s)))	#azimuth beam squint angle
    return sq_ang_1


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






def correct_channel(arr, az_spacing):

    ang_vec = _np.arange(arr.shape[1])
    sq_ang = squint_angle(arr.freq_vec, KU_WIDTH, KU_DZ)
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
    parser.add_argument('pat', type=str,
                help="Pattern to process")
    parser.add_argument('chan', type=str,
                help="Channel to process")
    #Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    #Read raw dataqset
    rawdata = _gpf.rawData(args.raw_par, args.raw)
    print(rawdata.az_spacing)
    #Empty dataset
    rawdata_corr = _np.zeros_like(rawdata) + rawdata
    #Channel index
    chan_idx = rawdata.channel_index(args.pat, args.chan)
    #Select channel of interest
    chan = rawdata[:,:, chan_idx[0], chan_idx[1]]
    #Apply interpolation
    chan_corr = correct_channel(chan, rawdata.az_spacing)
    #Write dataset
    with open(args.raw_out, 'wb') as of:
        chan_corr.T.astype(_gpf.type_mapping['SHORT INTEGER']).tofile(of)
    with open(args.raw_par, 'rt') as ip, open(args.raw_par_out, 'wt') as op:
        op.write(ip)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

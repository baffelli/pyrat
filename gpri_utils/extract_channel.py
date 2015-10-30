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



def main():
    #Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('raw',
                help="GPRI raw file")
    parser.add_argument('raw_par',
                help="GPRI raw file parameters")
    parser.add_argument('raw_out', type=str,
                help="GPRI raw file corresponding to the extracted channel")
    parser.add_argument('raw_par_out', type=str,
                help="GPRI raw parameter file corresponding to the extracted channel")
    parser.add_argument('pat', type=str,
                help="Pattern to extract")
    parser.add_argument('channel_mapping', type=str,
                help="File contanining the channel mapping informations. Format: TXA: pos, TXB: pos etc")
    #Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    #Read raw dataqset
    rawdata = _gpf.rawData(args.raw_par, args.raw)
    #Channel index
    try:
        chan_idx = rawdata.channel_index(args.pat[0:3],args.pat[3])
    except IndexError:
        Warning("This channel does not exist in the dataset")
    #Select channel of interest
    chan = rawdata[:,:, chan_idx[0], chan_idx[1]]
    #Empty dataset
    rawdata_out = chan
    #Write dataset
    with open(os.path.expanduser(args.raw_out), 'wb') as of:
        chan.T.astype(_gpf.type_mapping['SHORT INTEGER']).tofile(of)
    raw_dict = _gpf.par_to_dict(args.raw_par)
    if raw_dict['TX_mode'] == 'TX_RX_SEQ':
        pats =  raw_dict['TX_RX_SEQ'].split('-')
        npats = len(pats)
    else:
        npats = 1
    #Load channel mapping dictionary
    chan_mapping_dict = _gpf.par_to_dict(args.channel_mapping)
    #Construct keyword for antenna position dict
    tx_keyw = 'TX_{}_position'.format(args.pat[0])
    rx_keyw = 'RX_{}_position'.format(args.pat[2:4])
    #Extract antenna position
    raw_dict['GPRI_TX_antenna_position'] = chan_mapping_dict[tx_keyw]
    raw_dict['GPRI_RX_antenna_position'] = chan_mapping_dict[rx_keyw]
    raw_dict['ADC_capture_time'] = float(raw_dict['ADC_capture_time'] / npats)
    raw_dict['TSC_rotation_speed'] = raw_dict['TSC_rotation_speed'] * npats
    raw_dict['STP_rotation_speed'] = raw_dict['STP_rotation_speed'] * npats
    raw_dict['TSC_acc_ramp_time'] = raw_dict['TSC_acc_ramp_time'] / npats
    raw_dict['TX_mode'] = None
    raw_dict['TX_RX_SEQ'] = args.pat
    _gpf.dict_to_par(raw_dict, args.raw_par_out)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

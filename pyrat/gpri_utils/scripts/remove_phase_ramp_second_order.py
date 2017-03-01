#!/usr/bin/python

import argparse

import numpy as _np
import pyrat.fileutils.gpri_files as _gpf

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("phase", help="The path of the interferogram to flatten [FCOMPLEX]", type=str)
parser.add_argument('original_diff', help='The interferometric parameter file of the phase to correct',
                    type=str)
parser.add_argument("topo_phase_1", help="First topographic phase", type=str)
parser.add_argument('topo_diff_1', help='The interferometric parameter file of the first topographic phase',
                    type=str)
parser.add_argument("topo_phase_2", help="Second topographic phase", type=str)
parser.add_argument('topo_diff_2', help='The interferometric parameter file of the first topographic phase',
                    type=str)
parser.add_argument("x", help="The path of the interferogram to flatten [FCOMPLEX]", type=str)
parser.add_argument("y", help="The path of the interferogram to flatten [FCOMPLEX]", type=str)
parser.add_argument("z", help="The path of the interferogram to flatten [FCOMPLEX]", type=str)
args = parser.parse_args()

# Load phase containing topography
ph_pol_topo, ph_pol_topo_par = _gpf.load_dataset(args.original_diff, args.phase)
# Load topography 1 and 2
ph_topo_1, ph_topo_1_par = _gpf.load_dataset(args.topo_diff_1, args.topo_phase_1)
ph_topo_2, ph_topo_2_par = _gpf.load_dataset(args.topo_diff_2, args.topo_phase_2)
# Compute baselines and terms B
# Height of the GPRI w.r.t the ground
H = 1.7
B_pol = ph_pol_topo_par['antenna_separation'][0]
B_int_1 = ph_topo_1_par['antenna_separation'][0]
B_int_2 = ph_topo_2_par['antenna_separation'][0]


def aparam(dHtx1, dHrx1, dHtx2, dHrx2, H):
    return (dHtx1 - H) ** 2 - (dHtx2 - H) ** 2 + (dHrx1 - H) ** 2 - (dHrx2 - H) ** 2


# Determine terms x and y in the phase
A_int1 = aparam(ph_topo_1_par['image_1_transmitter_location'][2], ph_topo_1_par['image_1_receiver_location'][2],
                ph_topo_1_par['image_2_transmitter_location'][2], ph_topo_1_par['image_2_receiver_location'][2], H)
A_int2 = aparam(ph_topo_2_par['image_1_transmitter_location'][2], ph_topo_2_par['image_1_receiver_location'][2],
                ph_topo_2_par['image_2_transmitter_location'][2], ph_topo_2_par['image_2_receiver_location'][2], H)
A_pol = aparam(ph_pol_topo_par['image_1_transmitter_location'][2], ph_pol_topo_par['image_1_receiver_location'][2],
               ph_pol_topo_par['image_2_transmitter_location'][2], ph_pol_topo_par['image_2_receiver_location'][2], H)
# Solve the equation to compute the terms related to the first and second oreder expansion of the phase
# Compute the topographic phase at the polarimetric baseline
fc = 17.2e9
C = _gpf.C
lam = fc / C
x = 4 * _np.pi / lam * (A_int1 * ph_topo_2 - A_int2 * ph_topo_1) / (A_int1 * B_int_2 - A_int2 * B_int_1)
y = 4 * _np.pi / lam * (B_int_1 * ph_topo_2 - B_int_2 * ph_topo_1) / (A_int2 * B_int_1 - A_int1 * B_int_2)
topo_pol = _np.exp(-1j * (2 * B_pol * x + A_pol * y))
z = ph_pol_topo * topo_pol
with open(args.x, 'wb+') as of, open(args.y, 'wb+') as of1, open(args.z, 'wb+') as of2:
    x.astype(_gpf.type_mapping['FLOAT']).T.tofile(of)
    y.astype(_gpf.type_mapping['FLOAT']).T.tofile(of1)
    z.astype(_gpf.type_mapping['FCOMPLEX']).T.tofile(of2)

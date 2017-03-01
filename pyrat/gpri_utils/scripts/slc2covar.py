#!/usr/bin/python

import argparse

import numpy as _np
import pyrat.core.matrices as _mat
import pyrat.fileutils.gpri_files as _gpf

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('HH', help='HH slc', type=str)
parser.add_argument('HH_par', help='HH slc parameters', type=str)
parser.add_argument('HV', help='HV slc', type=str)
parser.add_argument('HV_par', help='HV slc parameters', type=str)
parser.add_argument('VH', help='HV slc', type=str)
parser.add_argument('VH_par', help='VH slc parameters', type=str)
parser.add_argument('VV', help='VV slc', type=str)
parser.add_argument('VV_par', help='VV slc parameters', type=str)
parser.add_argument('out_root', help='The root filename of the flattened covariance dataset')
# Two modes: remove or not
subparsers = parser.add_subparsers(help='Modes', dest='subparser_name')
subparser_remove = subparsers.add_parser('flatten', help='Remove topographic phase')
subparser_normal = subparsers.add_parser('noflatten', help='Do not remove topographic phase')
subparser_remove.add_argument("topo_phase",
                              help="Topographic phase (unwrapped float) with the same shape as the other SLCS",
                              type=str)
subparser_remove.add_argument("topo_phase_par", help="Topographic phase parameters",
                              type=str)
subparser_remove.add_argument("topo_phase_master_par",
                              help="SLC params of the master image used to generate the topographic phase", type=str)
subparser_remove.add_argument("topo_phase_slave_par",
                              help="SLC params of the slave image used to generate the topographic phase", type=str)

# Subparser to apply them
args = parser.parse_args()

# Load the channels
HH, HH_par = _gpf.load_dataset(args.HH_par, args.HH)
HV, HV_par = _gpf.load_dataset(args.HV_par, args.HV)
VH, VH_par = _gpf.load_dataset(args.VH_par, args.VH)
VV, VV_par = _gpf.load_dataset(args.VV_par, args.VV)

if args.subparser_name == 'flatten':
    # Load the topo phase
    topo_phase, topo_par = _gpf.load_dataset(args.topo_phase_par, args.topo_phase)

    # Load the master and slave slc parameters
    master_par = _gpf.par_to_dict(args.topo_phase_master_par)
    slave_par = _gpf.par_to_dict(args.topo_phase_slave_par)

    # Compute the two phase centers
    ph_center_topo_1 = _gpf.compute_phase_center(master_par)
    ph_center_topo_2 = _gpf.compute_phase_center(slave_par)
    int_bl = ph_center_topo_1 - ph_center_topo_2

# Construct the scattering matrix
C_matrix = _np.zeros(HH.shape + (4, 4), dtype=HH.dtype)
C_matrix_flat = _np.zeros(HH.shape + (4, 4), dtype=HH.dtype)
chan_list = [[HH, HH_par], [HV, HV_par], [VH, VH_par], [VV, VV_par]]
for idx_1, (chan_1, par_1) in enumerate(chan_list):
    for idx_2, (chan_2, par_2) in enumerate(chan_list):
        C_matrix[:, :, idx_1, idx_2] = chan_1 * chan_2.conj()
        if args.subparser_name == 'flatten':
            # Compute pase center of channel
            pol_ph_center_1 = _gpf.compute_phase_center(par_1)
            pol_ph_center_2 = _gpf.compute_phase_center(par_2)
            pol_bl = (pol_ph_center_1 - pol_ph_center_2)
            topo_phase_rescaled = _np.exp(1j * topo_phase * pol_bl / int_bl)
            C_matrix_flat[:, :, idx_1, idx_2] = C_matrix[:, :, idx_1, idx_2] * topo_phase_rescaled
        else:
            C_matrix_flat[:, :, idx_1, idx_2] = C_matrix[:, :, idx_1, idx_2]

C_matrix_flat = _mat.coherencyMatrix(C_matrix_flat, basis='lexicographic', bistatic=True)
C_matrix_flat.__dict__.update(HH_par)
C_matrix_flat.to_gamma(args.out_root, bistatic=True)

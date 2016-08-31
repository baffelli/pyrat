#!/usr/bin/python
__author__ = 'baffelli'
C = 299792458.0  # speed of light m/s
KU_WIDTH = 15.798e-3  # WG-62 Ku-Band waveguide width dimension
KU_DZ = 10.682e-3  # Ku-Band Waveguide slot spacing
RANGE_OFFSET = 3
ADCSR = 6250000.0
RF_CHIRP_RATE = 49942173063.8

import argparse
import sys

import numpy as _np
import pyrat.fileutils.gpri_files as _gpf
import scipy.signal as _sig


class gpriBackwardsProcessor:
    def __init__(self, args):
        # Load slc
        self.slc = _gpf.gammaDataset(args.slc_par, args.slc).astype(_np.complex64)
        self.raw_par_in = _gpf.par_to_dict(args.raw_par_in)
        self.slc_par = _gpf.par_to_dict(args.slc_par)
        self.args = args
        # Compute tcycle (supposing the decimation factor is known)
        self.tcycle = 1 / (self.slc_par['prf'][0]) / self.args.dec
        # Supposing a fixed ADC rate, the block length and the number of samples are easiyl computed
        self.block_length = int(self.tcycle * ADCSR) + 1
        self.nsamp = self.block_length - 1
        # We have to compute the chirp rate
        self.chirp_rate = self.slc_par['chirp_bandwidth'] / self.tcycle
        rps = (ADCSR / self.nsamp * C / 2.) / self.chirp_rate
        self.ns_min = int(round(args.rmin / rps)) * 2 + 1
        self.ns_max = int(round(args.rmax / rps)) * 2 + 1
        self.dt = _gpf.type_mapping[self.slc_par['image_format']]
        # Scale factor
        self.rps = (ADCSR / self.nsamp * C / 2.) / self.chirp_rate  # range pixel spacing
        self.pn1 = _np.arange(self.nsamp / 2 + 1)  # list of slant range pixel numbers
        self.slr = (
                   self.pn1 * ADCSR / self.nsamp * C / 2.) / self.chirp_rate + RANGE_OFFSET  # slant range for each sample
        self.scale = (abs(self.slr) / self.slr[self.nsamp / 8]) ** 1.5  # cubic range weighting in power
        self.ns_min = int(round(args.rmin / self.rps))  # round to the nearest range sample
        self.win = _sig.kaiser(self.nsamp, args.kbeta)
        if args.rmax != 0.0:  # check if greater than maximum value for the selected chirp
            if int(round(args.rmax / self.rps)) <= self.ns_max:
                self.ns_max = int(round(args.rmax / self.rps))
        else:
            self.ns_max = int(round(0.90 * self.nsamp / 2))
        self.ang_acc = self.raw_par_in['TSC_acc_ramp_angle']
        rate_max = self.raw_par_in['TSC_rotation_speed']
        self.t_acc = self.raw_par_in['TSC_acc_ramp_time']
        self.ang_per_tcycle = self.tcycle * self.raw_par_in['TSC_rotation_speed']  # angular sweep/transmit cycle
        self.nl_acc = int(self.t_acc / (self.tcycle * self.args.dec))
        self.win2 = _sig.hanning(
            2 * args.zero)
        self.win2[self.win2 == 0] = 1
        self.zero = args.zero

        #

    def decompress(self):
        arr_raw = _np.zeros((self.block_length, int(self.slc_par['azimuth_lines'] + 2 * self.nl_acc)),
                            dtype=_np.float32)
        fshift = _np.ones(self.nsamp / 2 + 1)
        fshift[0::2] = -1
        # Temporary "full" slc line
        full_line = _np.zeros(self.nsamp / 2 + 1, dtype=_np.complex64)
        # First convert back
        for idx_az in range(int(self.slc_par['azimuth_lines'])):
            load_line = self.slc[:, idx_az] * 1 / self.scale[self.ns_min:self.ns_max + 1]
            full_line[self.ns_min:self.ns_max + 1] = load_line
            arr_raw[1::, idx_az + self.nl_acc] = _np.fft.irfft(full_line * fshift, n=self.nsamp) / self.win
            if idx_az % 1000 == 0:
                out_str = "processing line {}".format(idx_az)
        if self.zero > 0:
            arr_raw[0:self.zero, :] = arr_raw[0:self.zero, :] / self.win2[0:self.zero, None]
            arr_raw[-self.zero:] = arr_raw[-self.zero:, :] / self.win2[-self.zero:, None]
        return arr_raw
        #
        # (.astype(_gpf.type_mapping['SHORT INTEGER'])).tofile(of)


def main():
    # Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('slc',
                        help="Slc file to process")
    parser.add_argument('slc_par',
                        help="GPRI slc file parameters")
    parser.add_argument('raw_par_in',
                        help="Input raw-par (used to compute acceleration time)")
    parser.add_argument('raw_out', type=str,
                        help="Output raw")
    parser.add_argument('raw_par_out', type=str,
                        help="raw parameter file")
    parser.add_argument('-d', type=int, default=1, dest='dec',
                        help="Decimation factor")
    parser.add_argument('-z',
                        help="Number of samples to zero at the beginning of each pulse", dest='zero',
                        type=int, default=300)
    parser.add_argument("-k", type=float, default=3.00, dest='kbeta',
                        help="Kaiser Window beta parameter")
    parser.add_argument("-s", "--apply_scale", type=bool, default=True, dest='apply_scale')
    parser.add_argument('-r', help='Near range for the slc', dest='rmin', type=float, default=0)
    parser.add_argument('-R', help='Far range for the slc', dest='rmax', type=float, default=1000)
    # Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    # Create processor object
    proc = gpriBackwardsProcessor(args)
    arr_raw = proc.decompress()
    arr_raw = arr_raw.astype(_gpf.type_mapping['SHORT INTEGER'])
    # (dataset, par_dict, par_file, bin_file)
    _gpf.write_dataset(arr_raw, proc.raw_par_in, args.raw_par_out, args.raw_out)
    # _gpf.dict_to_par(proc.raw_par_in, args.raw_par_out)
    # raw.T.astype(_gpf.type_mapping['SHORT INTEGER']).tofile(of)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

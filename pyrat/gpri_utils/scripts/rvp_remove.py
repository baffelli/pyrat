#!/usr/bin/python
__author__ = 'baffelli'
C = 299792458.0  # speed of light m/s
KU_WIDTH = 15.798e-3  # WG-62 Ku-Band waveguide width dimension
KU_DZ = 10.682e-3  # Ku-Band Waveguide slot spacing VV

RANGE_OFFSET = 3

import argparse
import sys

import numpy as _np
import pyrat.fileutils.gpri_files as _gpf
import scipy.signal as _sig


#
class gpriRVPProcessor:
    def __init__(self, args):
        # Pattern parameters
        # Load raw parameters and convert them into namedtuple
        self.rawdata, raw_par = _gpf.load_raw(args.raw_par, args.raw, nchan=1)
        print(self.rawdata.shape)
        self.raw_par = _gpf.rawParameters(raw_par, args.raw)
        self.fshift = _np.ones(self.raw_par.nsamp / 2 + 1)
        self.fshift[1::2] = -1
        self.rvp = _np.exp(1j * 4. * _np.pi * self.raw_par.grp.RF_chirp_rate * (
            self.raw_par.slr / C) ** 2)  # residual video phase correction
        # Fast (chirp) time
        self.fast_time = _np.arange(self.raw_par.nsamp) * 1 / self.raw_par.grp.ADC_sample_rate
        # Chirsp duration
        chirp_duration = 1 / self.raw_par.grp.ADC_sample_rate * self.raw_par.nsamp
        # Slow time
        self.slow_time = _np.linspace(0, self.raw_par.grp.ADC_capture_time, self.raw_par.nl_tot)
        # Range frequency
        self.range_freq = _np.linspace(self.raw_par.grp.RF_freq_min, self.raw_par.grp.RF_freq_max,
                                       self.raw_par.nsamp) - self.raw_par.grp.RF_center_freq
        print(self.range_freq)

    def correct(self):
        # First of all, correct Residual Video Phase
        self.rawdata_rvp = 1 * self.rawdata
        for idx_az in range(0, self.raw_par.nl_tot):
            temp = _np.append(_np.fft.ifftshift(
                _np.fft.irfft(_np.fft.rfft((self.rawdata[1:, idx_az].astype(_np.float32)) / 32768) * self.fshift *
                              self.rvp.astype('complex64').conj() * 32768), 0), 0)
            self.rawdata_rvp[:, idx_az] = temp
        # Then the data is transformed to IQ samples with an Hilbert transform
        rd_h = _sig.hilbert(self.rawdata_rvp, axis=0)
        rd_rmc = rd_h * 1
        # Compute rcmc
        # Antenna length
        r_ant = _np.sqrt(0.25 ** 2 + 0.15 ** 2)
        # rotation speed
        omega = _np.deg2rad(self.raw_par.grp.TSC_rotation_speed)
        # fmcw rate
        K = self.raw_par.grp.RF_chirp_rate
        # Wavelength
        lam = C / self.raw_par.grp.RF_center_freq
        # Reference slow time (taken at the center of the scene)
        t_slow_ref = self.slow_time[5000]
        # Reference fast time (taken at the center of the chirp)
        t_fast_ref = self.fast_time[0]
        # First rmc term
        first_term = 2 * r_ant / C * _np.cos((self.slow_time - t_slow_ref + t_fast_ref) * omega)
        second_term = -2 * r_ant * omega / (lam * K) * _np.sin(omega * (self.slow_time - t_slow_ref))
        rmc = _np.zeros(self.rawdata.shape, dtype=_np.complex64)
        for idx_f in range(1, rd_rmc.shape[0] - 1):
            rmc[idx_f, :] = _np.exp(-2j * _np.pi * (first_term + second_term) * self.range_freq[idx_f])
            rd_rmc[idx_f, :] = rd_h[idx_f, :] * rmc[idx_f, :]
        return rd_rmc.real, self.rawdata_rvp


def main():
    # Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('raw',
                        help="Raw channel file")
    parser.add_argument('raw_par',
                        help="GPRI raw file parameters")
    parser.add_argument('raw_out',
                        help="Raw channel file with RVP removed")
    parser.add_argument('raw_par_out',
                        help="GPRI raw file parameters")
    parser.add_argument('raw_out_rmc',
                        help="Raw channel file with RVP removed and RMC correction")
    parser.add_argument('raw_par_out_rmc',
                        help="GPRI raw file parameters")
    # Read adrguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    # Create processor object
    proc = gpriRVPProcessor(args)
    raw_rmc, raw_rvp = proc.correct()
    with open(args.raw_out, 'wb') as of:
        raw_rvp.astype(_gpf.type_mapping['SHORT INTEGER']).T.tofile(of)
    with open(args.raw_out_rmc, 'wb') as of:
        raw_rmc.astype(_gpf.type_mapping['SHORT INTEGER']).T.tofile(of)
    _gpf.dict_to_par(_gpf.par_to_dict(args.raw_par), args.raw_par_out)
    _gpf.dict_to_par(_gpf.par_to_dict(args.raw_par), args.raw_par_out_rmc)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

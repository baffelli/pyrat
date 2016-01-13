#!/usr/bin/python
__author__ = 'baffelli'
import argparse
import sys
import os
import numpy as _np
import scipy as _sp
import scipy.signal as _sig
import simulate_squint as sq
import pyrat.fileutils.gpri_files as _gpf
import matplotlib.pyplot as _plt
class rawTracker:

    def __init__(self, args):
        self.args = args
        #Load raw file parameters
        raw_par = _gpf.par_to_dict(args.raw_par)
        self.grp = _gpf.rawParameters(raw_par, self.args.raw)
        self.raw_data = []
        self.fshift = _np.ones(self.grp.nsamp/ 2 +1)
        self.fshift[1::2] = -1


    def track_maximum(self):
        #Start record between zero and the total number of lines
        az_start_idx =  _np.clip(self.args.aztarg - self.args.search_window/2,
                                 0,self.grp.nl_tot)
        self.raw_data = _gpf.load_segment(self.args.raw,
                                          (self.grp.block_length, self.grp.nl_tot)
                                          ,0, self.grp.block_length,
                                          az_start_idx,
                                          _np.clip(self.args.aztarg + self.args.search_window/2,0,self.grp.nl_tot),
                                          dtype=_gpf.type_mapping['SHORT INTEGER'])
        az_min = self.grp.grp.STP_antenna_start + self.grp.ang_per_tcycle * az_start_idx
        #azimuth indices
        az_vec = az_min + self.grp.ang_per_tcycle * _np.arange(self.raw_data.shape[1])
        #First we compute the range spectrum
        range_spectrum = _np.fft.rfft(self.raw_data,axis=0) * self.fshift[:,None] * self.grp.scale[:,None]
        #From the range spectrum we can obtain the maximum
        #From the maximum energy we determine the dominant target frequency
        max_idx = _np.argmax(range_spectrum[1:,:])
        range_spectrum_shape = [range_spectrum.shape[0]- 1, range_spectrum.shape[1]]
        max_r, max_az = _np.unravel_index(max_idx,range_spectrum_shape)
        freq_vec = _np.fft.rfftfreq(range_spectrum.shape[0])
        ws = 100
        freq_filter = _np.zeros(range_spectrum_shape[0] + 1)
        freq_filter[max_r-ws/2:max_r+ws/2] = _np.hamming(ws)
        #Filter spectrum to extract only range of interest
        filt_data = _np.fft.irfft(range_spectrum * freq_filter[:,None] * -self.fshift[:,None],axis=0)
        #COmpute its envelope
        filt_env = _sig.hilbert(filt_data,axis=0)
        #Track the maximum and convert the indices into angles
        max_vec = _np.array( [_np.argmax(_np.abs(filt_env[idx,:]))
                                 for idx in range(filt_env.shape[0])])
        # squint_vec = az_vec[max_vec]
        az_vec_plot = az_vec - az_vec[max_vec[max_vec.shape[0]/2]]
        squint_vec = az_vec_plot[max_vec]
        #Fit a polynomial
        start_idx = 300
        pars = _sp.polyfit(self.grp.freq_vec[start_idx:-start_idx],squint_vec[start_idx:-start_idx],1)
        squint_vec_fit = _np.polynomial.polynomial.polyval(self.grp.freq_vec, pars[::-1])
        print(pars)
        pars[-1] = 0
        #Compute squint with simulation
        squint_vec_sim = sq.squint_angle(self.grp.freq_vec, chan='H')
        #Add the location of estimated squint at 0
        squint_vec_sim = pars[-1] + squint_vec_sim - squint_vec_sim[squint_vec_sim.shape[0]/2]
        #Computed fitted data
        #Use specified style sheet for the plots
        if self.args.style is not '':
            sty = 'classic'
        else:
            sty = self.args.style
        with _plt.style.context(sty):
            f = _plt.figure()
            _plt.plot(self.grp.freq_vec[start_idx:-start_idx],squint_vec[start_idx:-start_idx], label=r'measured')
            _plt.plot(self.grp.freq_vec, squint_vec_fit, label=r' Linear model')
            # _plt.plot(self.grp.freq_vec,squint_vec_sim, label=r' Exact Formula')
            _plt.xlabel(r'Chirp Frequency [Hz]')
            _plt.ylabel(r'Azimuth Offset [deg]')
            _plt.grid()
            _plt.legend()
            _plt.show()
            f.savefig(self.args.squint_plot)
             #Computed fitted data
            f, ax  = _plt.subplots()
            #line width
            lw = 3
            # ax.plot(squint_vec[start_idx:-start_idx], self.grp.freq_vec[start_idx:-start_idx], label='measured')
            freq_vec_plot = self.grp.freq_vec / 1e9
            ax.plot(squint_vec_fit,freq_vec_plot, label=r' Linear model', lw=lw)
            ax.plot(squint_vec_sim,freq_vec_plot, label=r' Exact model', lw=lw)
            ax.imshow(_np.abs(filt_env)**0.3,
                        extent=[az_vec_plot[0], az_vec_plot[-1],freq_vec_plot[0],freq_vec_plot[-1],],
                      aspect=1e2, alpha=0.8, origin='lower', interpolation='none')

            _plt.ylabel(r'Chirp Frequency [GHz]')
            _plt.xlabel(r'Angle from Antenna Pointing At $f_{c}$[deg]')
            _plt.title(r'Squint parameters: ${:3.2e}f$'.format(*pars), fontsize=15)
            _plt.grid()
            _plt.legend()
            f.tight_layout()
            _plt.show()
            f.savefig(self.args.squint_image)

        #fitted_squint = pars[0] * self.grp.freq_vec + pars[1]
        return _np.transpose([self.grp.freq_vec,squint_vec]), pars



def main():
    #Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('raw',
                help="Raw file to process")
    parser.add_argument('raw_par',
                help="Raw file parameters to process")
    parser.add_argument('aztarg', type=int,
                help="Target of interest azimuth location")
    parser.add_argument('squint', type=str,
                help="Output squint profile (text)")
    parser.add_argument('squint_fit', type=str,
                help="Linear squint parameters (text)")
    parser.add_argument('--search-window', dest='search_window', default=200, type=int, )
    parser.add_argument('--squint_plot', default='', type=str, )
    parser.add_argument('--squint_image', default='', type=str, )
    parser.add_argument('--style', default='', type=str, help='Matplotlib Style file')

    #Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    #Create processor object
    proc = rawTracker(args)
    squint_data, squint_par = proc.track_maximum()
    with open(args.squint, 'w') as of, open(args.squint_fit, 'w') as of1:
        _np.savetxt(of,squint_data,header="Frequency\tSquint")
        _np.savetxt(of1,squint_par)
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

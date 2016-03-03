#!/usr/bin/python
__author__ = 'baffelli'
import argparse
import sys
import os
import numpy as _np
import scipy as _sp
import scipy.signal as _sig
import simulate_squint as _sq
import pyrat.fileutils.gpri_files as _gpf
import pyrat.visualization.visfun as _vf
import matplotlib.pyplot as _plt


def stft(x, fftsize=4096, overlap=50):
    hop = fftsize / overlap
    w = _sp.hanning(fftsize+1)[:-1]      # better reconstruction with this trick +1)[:-1]
    fshift = _np.ones(fftsize/2+1)
    fshift[1::2] = -1
    return _np.array([_np.fft.rfft(w[:,None]*x[i:i+fftsize,1:],axis=0)*fshift[:,None] for i in range(0, x.shape[0]-fftsize, hop)])


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
        az_start_idx =  _np.clip(self.args.aztarg - self.args.search_window/2.0,
                                 0,self.grp.nl_tot)
        az_stop_idx = _np.clip(self.args.aztarg + self.args.search_window/2.0,0,self.grp.nl_tot)
        self.raw_data = _gpf.load_segment(self.args.raw,
                                          (self.grp.block_length, self.grp.nl_tot)
                                          ,0, self.grp.block_length,
                                          az_start_idx, az_stop_idx,
                                          dtype=_gpf.type_mapping['SHORT INTEGER']).astype(_np.float32) / 32768
        az_min = self.grp.grp.STP_antenna_start + self.grp.ang_per_tcycle * az_start_idx
        #First we compress to a mini-slc
        range_spectrum = _np.fft.rfft(self.raw_data[1:,:],axis=0) * self.fshift[:,None] * self.grp.scale[:,None]
        #range_spectrum *=  _np.hamming(range_spectrum.shape[1])[None,:]
        #From the range spectrum we can obtain the maximum
        #From the maximum energy we determine the dominant target frequency
        max_idx = _np.argmax(_np.abs(range_spectrum))#we subtract 1 because the first sample is used to jump
        range_spectrum_shape = [range_spectrum.shape[0], range_spectrum.shape[1]]
        max_r, max_az = _np.unravel_index(max_idx,range_spectrum_shape)
        _plt.figure()
        _plt.imshow(_np.abs(range_spectrum))
        _plt.plot(max_az,max_r,'ro')
        _plt.show()
        #We compute the corresponding filter, that is the beat frequency corresponding to
        #the range of the reflector
        r_max = self.grp.slr[max_r]
        t_chirp = self.grp.grp.CHP_num_samp/self.grp.grp.ADC_sample_rate
        bw = (self.grp.grp.RF_freq_max - self.grp.grp.RF_freq_min)
        range_freq = 4 * _np.pi * 2 * r_max * bw / float(_gpf.C * t_chirp)
        range_filter = _np.exp(1j * range_freq * _np.arange(0,self.grp.block_length - 1)/self.grp.grp.ADC_sample_rate)
        # #We transform the data into hilbert domain (analytic signal)
        # chirp_analytic = _sig.hilbert(self.raw_data[1:,:], axis = 0)
        #Cut the data around the maxium in azimuth
        range_spectrum = range_spectrum[:, max_az - self.args.analysis_window/2:max_az + self.args.analysis_window/2]
        #azimuth indices
        az_vec = (max_az - self.args.analysis_window/2) * self.grp.ang_per_tcycle + az_min + self.grp.ang_per_tcycle * _np.arange(range_spectrum.shape[1])
        #Window size for the filter
        freq_filter = _np.zeros(range_spectrum_shape[0])
        #Window position
        freq_filter[max_r-self.args.range_window/2:max_r+self.args.range_window/2] = _np.hamming(self.args.range_window)
        #Filter spectrum to extract only range of interest
        filt_data = _np.fft.irfft(range_spectrum * freq_filter[:,None] * -self.fshift[:,None],axis=0)
        # filt_data = self.raw_data[1:, max_az - self.args.analysis_window/2:max_az + self.args.analysis_window/2] * _np.real(range_filter[:,None])
        #COmpute its envelope
        filt_env = _sig.hilbert(filt_data,axis=0)
        _plt.imshow(_np.abs(filt_env))
        _plt.show()
        #Track the maximum and convert the indices into angles
        max_vec = _np.array( [_np.argmax(_np.abs(filt_env[idx,:]))
                                 for idx in range(filt_env.shape[0])])
        #Extract the envelope at the position of the maximum
        phase_response =_np.array( [filt_env[idx,max_vec[idx]] for idx in range(filt_env.shape[0])])
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
        squint_vec_sim = _sq.squint_angle(self.grp.freq_vec, chan='H', k=1/2.0)
        #Add the location of estimated squint at 0
        squint_vec_sim = pars[-1] + squint_vec_sim - squint_vec_sim[squint_vec_sim.shape[0]/2]
        #Compress the filtered spectrum to obtain image around reflector
        compressed_spectrum = _np.fft.rfft(filt_data,axis=0)* self.fshift[:,None] * self.grp.scale[:,None]
        if self.args.squint_image:
            #Use specified style sheet for the plots
            if self.args.style is '':
                sty = 'classic'
            else:
                sty = self.args.style
            with _plt.style.context(sty):
                f, ax  = _plt.subplots()
                #line width
                lw = 3
                freq_vec_plot = self.grp.freq_vec / 1e9
                ax.plot(freq_vec_plot,squint_vec_fit, label=r' Linear fit $\phi = \alpha f$', lw=lw)
                if self.args.plot_model:
                    ax.plot(freq_vec_plot,squint_vec_sim, label=r' Theoretical model', lw=lw)
                phase_amp, pal, norm = _vf.dismph(filt_env.T,k=1.2)
                ax.imshow(_np.abs(filt_env.T),
                            extent=[freq_vec_plot[0],freq_vec_plot[-1],az_vec_plot[0], az_vec_plot[-1]],
                          aspect=1/50.0, alpha=0.8, origin='lower', interpolation='none')
                _plt.xlabel(r'Chirp Frequency [GHz]')
                _plt.ylabel(r'Offset from pointing at $f_{c}$[deg]')
                _plt.title(r'Squint parameters: ${:3.2e}f$'.format(*pars), fontsize=15)
                _plt.grid()
                _plt.legend(loc=0)
                #f.tight_layout()
                _plt.show()
                f.savefig(self.args.squint_image,bbox_inches='tight')
        #Print squint at fc for VV and HH
        fc = self.grp.freq_vec[self.grp.freq_vec.shape[0]/2]
        #Modeled squint for HH and VV
        squint_VV = _sq.squint_angle(fc,chan='V', k=1/2.0)
        squint_HH = _sq.squint_angle(fc,chan='V', k=1/2.0)
        print('HH initial squint: {} \n VV initial squint: {}'.format(squint_HH,squint_VV))
        print(_sq.KU_DZ['V']/_sq.lam(17.2e9))
        print(_sq.lamg(17.2e9, _sq.KU_WIDTH))
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
    parser.add_argument('--search-window', dest='search_window', default=200, type=int, help='Search window to look for the maxium response' )
    parser.add_argument('--analysis-window', dest='analysis_window', default=50, type=int, help='Azimuth Window for the anaylsis [samples]' )
    parser.add_argument('--range-window', dest='range_window', default=100, type=int, help='Range window for the spectral anaylsis [samples]' )
    parser.add_argument('--squint_plot', default='', type=str, )
    parser.add_argument('--squint_image', default='', type=str, )
    parser.add_argument('--style', default='', type=str, help='Matplotlib Style file')
    parser.add_argument('--plot_model', default=False, action='store_true', help='Plot curve obtained from model?')

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

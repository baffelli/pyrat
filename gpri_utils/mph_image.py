#!/usr/bin/python
__author__ = 'baffelli'


import sys, os
import numpy as _np
import argparse
sys.path.append(os.path.expanduser('~/PhD/trunk/Code/'))
import pyrat.visualization.visfun as _vf
import pyrat.fileutils.gpri_files as _gpf
import matplotlib.pyplot as _plt
import matplotlib as _mpl
from matplotlib import style as _sty




class dismphPlotter:

    def __init__(self, args):
        #Determine how many bits to load
        fs = os.path.getsize(args.image)
        shape = [args.width, fs / _gpf.type_mapping['FCOMPLEX'].itemsize]
        #Number of elements
        #Load image
        self.image = _np.fromfile(args.image, dtype= _gpf.type_mapping['FCOMPLEX'])
        #Handling defaults
        if args.nlines is '-' or '0':
            args.nlines = args.width
        else:
            args.nlines = int(args.nlines)
        if args.start is '-':
            args.start = 0
        else:
            args.start = int(args.start)
        if args.scale is '-':
            args.scale = 1
        else:
            args.scale = int(args.scale)
        if args.k is '-':
            args.k = 0.35
        else:
            args.k = int(args.k)
        #Load style
        if args.style is None:
            import pyrat as _pt
            path = os.path.dirname(_pt.__file__)
            self.style = path + '/paper_rc.rc'
        else:
            self.style = args.style
        self.args = args

    def display(self):
        with _sty.use(self.style):
            f = _plt.figure()
            RGB, pal = _vf.dismph(self.image, k = self.args.k)
            _plt.imshow(_RGB)
            _plt.show()
            f.savefig(self.args.figpath)

def main():
    #Read the arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('image', type=str,
                help="Fcomplex image")
    parser.add_argument('width', type=str,
                help="Width of image (complex samples per line)")
    parser.add_argument('figpath', type=str,
                help="Path to save the output image")
    parser.add_argument('start', type=str, default='0',
                        help='First line to display')
    parser.add_argument('nlines', type=str, default='0',
                        help='Number of lines to display')
    parser.add_argument('scale', type=str, default='1',
                         help='Display scale factor')
    parser.add_argument('k', type=str
                        , default='1', help=' Display exponent')
    parser.add_argument('-l', '--labels', dest='labels', type=str, nargs=2, default=[None, None],
                help="Labels to display in the axes (default: none displayed)")
    parser.add_argument('-s', '--style', dest='style',
                        help="Matplotlibrc style file", default=None)
    #Read arguments
    try:
        args = parser.parse_args()
    except:
        print(parser.print_help())
        sys.exit(-1)
    #Create processor object
    proc = dismphPlotter(args)
    slc_dec = proc.display()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

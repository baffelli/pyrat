from unittest import TestCase
import numpy as np
from .. import corefun
import matplotlib.pyplot as plt
import pyrat.visualization.visfun as vf


class TestComplex_interpolate(TestCase):
    def setUp(self):
        xx, yy = np.mgrid[0:50,0:50]
        xx -= xx[xx.shape[0]//2,xx.shape[1]//2]
        yy -= yy[yy.shape[0]//2,yy.shape[1]//2]

        amp = np.exp(-0.02*(xx**2+yy*2))
        ph = np.exp(1j*0.2*(2 * xx + 3 * np.sin(yy)))
        self.im = ph * amp

    def testCompareInterp(self):
        osf = 4
        int_cart = corefun.complex_interp(self.im, osf, polar=False)
        int_pol = corefun.complex_interp(self.im, osf, polar=True)
        rgb_orig, *rest = vf.dismph(self.im)
        rgb_pol, *rest = vf.dismph(int_pol)
        rgb_cart, *rest = vf.dismph(int_cart)
        f = plt.figure()
        orig_ax = f.add_subplot(3,1,1)
        orig_ax.imshow(rgb_orig)
        orig_ax.imshow(rgb_orig)
        pol_ax = f.add_subplot(3, 1, 2)
        cart_ax = f.add_subplot(3, 1, 3,sharex=pol_ax, sharey=pol_ax)
        cart_ax.imshow(rgb_cart)
        orig_ax.set_title('Original')
        cart_ax.set_title('Cartesian')
        pol_ax.set_title('Polar')
        plt.show()

KU_WIDTH = 15.798e-3 #WG-62 Ku-Band waveguide width dimension
KU_DZ = 10.682e-3   #Ku-Band Waveguide slot spacing
#KU_DZ = 10.3e-3   #Ku-Band Waveguide slot spacing
C = 299792458.0    #speed of light m/s
import numpy as _np


#Lambda in waweguide
def lamg(freq, w):
    la = lam(freq)
    return la / _np.sqrt(1.0 - (la / (2 * w)) ** 2)	#wavelength in WG-62 waveguide

#lambda in freespace
def lam(freq):
    return C/freq


def squint_angle(freq, w, s):
    sq_ang = _np.arccos(lam(freq) / lamg(freq, w) - lam(freq) / (2 * s))
    dphi = _np.pi * (2. * KU_DZ / lamg(freq, w) - 1.0)				#antenna phase taper to generate squint
    sq_ang_1 = (_np.arcsin(lam(freq) * dphi / (2. * _np.pi * s)))	#azimuth beam squint angle
    return sq_ang_1


# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 11:25:00 2014

@author: baffelli
"""
import numpy as _np

from ..core import corefun as _cf


"""
Pyrat module for interferometric processing
"""

def F_model(dt):
    #F matrix for variable dt
    F_m = _np.array([[1, dt], [0, 1]])
    return F_m

def H_stack(f_fun, H_model, itab, t_vector):
    """
    Constructs a "H" output matrix for a
    linear system representing a stack of interferograms
    generated with `itab` and delta-timmes taken from
    t_vector
    Parameters
    ----------
    f_fun : function
        Function to generate the state transition matrix as a function of the timestep
    H_model : np.ndarray
        Output matrix for the linear displacement model
    itab : Itab
        Itab, containing the pairs of slcs to compute interferograms for
    t_vector : list
        list of acquisition times

    Returns
    -------

    """
    F_aug = []
    A = itab.to_incidence_matrix()
    F = _np.eye(2)
    t_start  = t_vector[0]
    for idx_t, t in enumerate(t_vector[1::]):
        t_end = t_vector[idx_t]
        dt = t_end - t_start
        F_model = f_fun(dt)
        F = _np.dot(F_model,F)
        F_aug.append(_np.dot(H_model, F))
        t_start = t_end
    F_aug = _np.vstack(F_aug)
    pi = _np.dot(A, F_aug)
    return pi

def estimate_coherence(ifgram, mli1, mli2, win, discard=True):
    """
    Estimates the coherence from a complex interferogram
    and two intensity images
    Parameters
    ----------
    ifgram
    mli1
    mli2

    Returns
    -------

    """
    cc = _cf.smooth(ifgram, win, discard=discard) / _np.sqrt((_cf.smooth(mli1, win, discard=discard) * _cf.smooth(mli2, win, discard=discard)))
    return cc

def compute_baseline(slc1, slc2):
    bl = slc1.phase_center - slc2.phase_center

class Itab:
    """
    Class to represent itab files, list of acquisitions to compute
    interferograms
    """
    def __init__(self, n_slc, stride=1, window=None, step=1, n_ref=0, **kwargs):
        tab = []
        self.n_slc = n_slc
        self.stride = stride
        self.window = window or n_slc
        self.step = step
        self.n_ref = 0
        #list with reference numbers
        if stride == 0:
            reference = [n_ref]
            window = 0
        else:
            reference = list(range(n_ref,n_slc, stride))
        counter = 1
        for master in reference:
            for slave in range(master + step, master+ step+window, step):
                if slave < n_slc:
                    line = [master, slave, counter]
                    counter += 1
                    tab.append(line)
        self.tab = tab

    def tofile(self, file):
        with open(file, 'w+') as of:
            for line in self.tab:
                of.writelines(" ".join(map(str, line)) + " 1" + '\n')


    def to_incidence_matrix(self):
        n_slc = self.n_slc
        A = _np.zeros((len(self.tab),n_slc+1))
        for idx_master, idx_slave, idx_itab, *rest in self.tab:
            A[idx_itab - 1, idx_master] = 1
            A[idx_itab - 1, idx_slave] = -1
        return A
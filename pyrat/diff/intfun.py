# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 11:25:00 2014

@author: baffelli
"""
import numpy as _np
import scipy as _sp

from ..core import corefun as _cf

"""
Pyrat module for interferometric processing
"""

def F_model(dt, order=1):
    #F matrix for variable dt\
    #first line
    # first_line = [dt**order/(order) for order in order]
    #
    # F_m = _np.eye(order)
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
    itab : pyrat.diff.utils.Itab
        Itab, containing the pairs of slcs to compute interferograms for
    t_vector : list
        list of acquisition times

    Returns
    -------

    """
    F_aug = []
    A = itab.to_incidence_matrix()
    F = _np.eye(2)
    t_start = t_vector[0]
    for t in t_vector[1::]:
        dt = t_start - t
        F_model = f_fun(dt)
        F = _np.dot(F_model,F)
        F_aug.append(F)
        t_start = t
    H_aug = _sp.linalg.block_diag(*[H_model,]*len(F_aug))
    F_aug = _np.vstack(F_aug)
    print(F_aug)
    pi = _np.dot(A, _np.dot(H_aug,F_aug))
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


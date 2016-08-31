# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 16:37:53 2014

@author: baffelli

Function to operate on stacks of images
"""
import numpy as np
import gpri_utils.calibration as calibration
class stack:
    
    def __init__(self,stack):
        try:
            iterator = iter(stack)
        except TypeError:
            pass
        else:
            self.stack = stack
            
    def __iter__(self):
        return iter(self.stack)
    
    def __getitem__(self, key):
        return self.stack.__getitem__(key)


    def scattering_to_coherency(self, win = None, **kwargs):
        """
        Converts a stack of scattering matrices into a stack of coherency matrices
        Parameters
        ----------
        win : array_like
            The averaging window
        """
        coherency_stack = []
        for s in self:
            try:
                T = s.to_coherency_matrix(**kwargs)
            except AttributeError:
                raise AttributeError('The current stack cannot be converted to coherency matrices')
            if win is not None:
                T = T.boxcar_filter(win)
            else:
                pass
            coherency_stack.append(T)
        return stack(coherency_stack)
    
    def calibrate_stack(self, parameter_path):
        cal_stack = []
        for s in self:
            try:
                s_cal = calibration.calibrate_from_parameters(s, parameter_path)
            except:
                raise ValueError('The current stack has not the correct format for calibration')
            cal_stack.append(s_cal)
        return stack(cal_stack)

    def cloude_pottier(self):
        entropy_stack = []
        anisotropy_stack = []
        alpha_stack = []
        beta_stack = []
        for s in self:
            try:
                entropy, anisotropy, alpha, beta, p, w = s.cloude_pottier()
            except AttributeError:
                raise AttributeError('The current stack does not consist of coherency matrices')
            entropy_stack.append(entropy)
            anisotropy_stack.append(anisotropy)
            alpha_stack.append(alpha)
            beta_stack.append(beta)
        entropy = np.dstack(entropy_stack)
        anisotropy = np.dstack(anisotropy_stack)
        alpha = np.dstack(alpha_stack)
        beta = np.dstack(beta_stack)
        return entropy, anisotropy, alpha, beta
        
import pyrat.diff.utils

from ..fileutils import gpri_files as gpf

import scipy as _sp

import datetime as _dt

from . import intfun

import numpy as _np

class Interferogram(gpf.gammaDataset):
    """
    Class to represent a complex interferogram
    """

    def __new__(cls, *args, **kwargs):
        ifgram, ifgram_par = gpf.load_dataset(args[0], args[1], **kwargs)
        ifgram = ifgram.view(cls)
        ifgram._params = ifgram_par.copy()
        #Add properties of master and slave to the dict by adding them and appending
        if "master_par" in kwargs:
            master_par_path = kwargs.pop('master_par')
            slave_par_path = kwargs.pop('slave_par')
            ifgram.master_par = gpf.par_to_dict(master_par_path)
            ifgram.slave_par = gpf.par_to_dict(slave_par_path)
            ifgram.add_parameter('slave_par', slave_par_path)
            ifgram.add_parameter('master_par', master_par_path)
        elif "slave_par" in ifgram._params:
            ifgram.slave_par = gpf.par_to_dict(ifgram._params.slave_par)
            ifgram.master_par = gpf.par_to_dict(ifgram._params.master_par)
        return ifgram

    @property
    def master_time(self):
        return gpf.datetime_from_par_dict(self.master_par)

    @property
    def slave_time(self):
        return gpf.datetime_from_par_dict(self.master_par)

    @property
    def temporal_baseline(self):
        return self.master_par.start_time - self.slave_par.start_time

    def tofile(self, par_path, bin_path):
        arr = self.astype(gpf.type_mapping['FCOMPLEX'])
        gpf.write_dataset(arr, self._params, par_path, bin_path)



class Stack:
    """
    Class to represent a stack of interferograms
    """
    def __init__(self, par_list, bin_list, mli_par_list, itab, *args, **kwargs):
        stack = []
        #load mli parameters
        mli_pars = [gpf.par_to_dict(f) for f in mli_par_list]
        for par_name, bin_name in zip(par_list,bin_list):
            ifgram = Interferogram(par_name, bin_name, **kwargs)
            stack.append(ifgram)
        #Sort by acquisition time
        sorting_key = lambda x: (x.master_time, x.slave_time)
        stack = sorted(stack, key=sorting_key )
        self.stack  = stack#the stack
        self.itab = pyrat.diff.utils.Itab.fromfile(itab)#the corresponding itab file
        self.slc_tab = sorted(mli_pars, key=lambda x: [(x.date, x.start_time)])

    @property
    def dt(self):
        return [s.temporal_baseline for s in self.stack]

    def __getitem__(self, item):
        return self.stack.__getitem__(item)

    def H_stack(self, f_fun, H_model):
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
        A = self.itab.to_incidence_matrix()
        F = _np.eye(2)
        t_vec = [t.start_time for t in self.slc_tab]
        t_start = t_vec[0]
        for t in t_vec[::]:
            dt = t - t_start
            F_model = f_fun(dt)
            F = _np.dot(F_model, F)
            F_aug.append(F)
        H_aug = _sp.linalg.block_diag(*[H_model,] * len(F_aug))
        F_aug = _np.vstack(F_aug)
        out_aug =  _np.dot(H_aug, F_aug)
        pi = _np.dot(A, out_aug)
        return pi
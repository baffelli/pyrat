import itertools as _iter
import pickle

import numpy as _np
import pyrat.diff.utils
import scipy as _sp
import scipy.misc as _misc

from ..fileutils import gpri_files as gpf

import numpy.ma as _ma

class Interferogram(gpf.gammaDataset):
    """
    Class to represent a complex interferogram
    """

    def __new__(cls, *args, **kwargs):
        ifgram, ifgram_par = gpf.load_dataset(args[0], args[1], **kwargs)
        ifgram = ifgram.view(cls)
        ifgram._params = ifgram_par.copy()
        # Add properties of master and slave to the dict by adding them and appending
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

    def __init__(self, par_list, bin_list, mli_par_list, itab, cc=None, mask=None, *args, **kwargs):
        stack = []
        cc_stack = []
        mask_stack = []
        # load mli parameters
        mli_pars = [gpf.par_to_dict(f) for f in mli_par_list]
        for idx, par_name, bin_name, cc_name, mask_name in enumerate(
                _iter.zip_longest(par_list, bin_list, cc, mask)):
            ifgram = Interferogram(par_name, bin_name, **kwargs)
            stack.append(ifgram)
            if cc is not None:
                cc_stack.append(gpf.load_binary(cc_name), ifgram.shape[0])
            if mask is not None:
                mask_stack.append(_misc.imread(input.unw_mask, mode='L'))

        # Sort by acquisition time
        sorting_key = lambda x: (x.master_time, x.slave_time)
        stack = sorted(stack, key=sorting_key)
        self.stack = stack  # the stack
        self.itab = pyrat.diff.utils.Itab.fromfile(itab)  # the corresponding itab file
        self.slc_tab = sorted(mli_pars, key=lambda x: [(x.date, x.start_time)])
        if cc is not None:
            self.cc = cc_stack
        if mask is not None:
            self.mask = mask

    @classmethod
    def fromfile(cls, file):
        return pickle.load(file)

    def tofile(self, file):
        pickle.dump(file, protocol=0)

    def flatten(self):
        """
        Flattens the stack: all the pixels will
        be stored in a single 1d array. The shape
        will be `(npixels, nstack)`

        Returns
        -------
        :obj:`np.ndarray`

        """

    @property
    def dt(self):
        return [s.temporal_baseline for s in self.stack]

    def __getitem__(self, item):
        """
        Get layer in stack and return it as a masked array, where
        the not unwrapped pixels are masked
        Parameters
        ----------
        item

        Returns
        -------

        """
        current_mask = self.mask.__getitem__(item)
        return _ma.masked_array(data=self.stack.__getitem__(item), mask=current_mask)

    def R_stack(self):
        """
        Constructs the interferogram covariance matrix
        Returns
        -------

        """

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
        H_aug = _sp.linalg.block_diag(*[H_model, ] * len(F_aug))
        F_aug = _np.vstack(F_aug)
        out_aug = _np.dot(H_aug, F_aug)
        pi = _np.dot(A, out_aug)
        return pi

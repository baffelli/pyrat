# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:20:32 2014

@author: baffelli
"""
import copy as _cp

import numpy as _np

from . import corefun
from ..fileutils import gpri_files, other_files
from ..visualization import visfun


def gpi(input_mat, basis='pauli', **kwargs):
    if type(input_mat) is scatteringMatrix:
        k = _np.abs(input_mat.scattering_vector(basis=basis))[:, :, [1, 2, 0]]
    else:
        if input_mat.basis is 'pauli':
            k = _np.diagonal(input_mat, axis1=2, axis2=3)[:, :, [1, 2, 0]]
        else:
            k = _np.diagonal(input_mat, axis1=2, axis2=3)
            if input_mat.shape[-1] == 4:
                k = k[:, :, [0, 1, 3]]
    im = visfun.pauli_rgb(k, **kwargs)
    return im


# Define shared methods
def __law__(input_obj, out_obj):
    # This is just a lazy
    # _array_wrap_ that
    # copies properties from
    # the input object
    out_obj = out_obj.view(type(input_obj))
    try:
        out_obj.__dict__ = input_obj.__dict__.copy()
    except:
        pass
    return out_obj


def __laf__(self_obj, obj):
    if obj is None: return
    if hasattr(obj, '__dict__'):
        obj.__dict__.update(self_obj.__dict__)


def __general__getitem__(obj, sl_in):
    # Contstruct indices
    if type(sl_in) is str:
        try:
            sl_mat = gpri_files.channel_dict[sl_in]
            sl = (Ellipsis,) * (obj.ndim - 2) + sl_mat
        except KeyError:
            raise IndexError('This channel does not exist')
    else:
        sl = sl_in
    # Apply slicing
    new_obj_1 = obj.__array_wrap__(_np.array(obj).__getitem__(sl))
    # Update attributes
    if hasattr(obj, 'r_vec'):
        try:
            r_vec = obj.r_vec[sl[1]]
            az_vec = obj.az_vec[sl[0]]
            new_obj_1.__setattr__('r_vec', r_vec)
            new_obj_1.__setattr__('az_vec', az_vec)
        except:
            pass
    return new_obj_1


def __general__setitem__(obj, sl_in, item):
    obj1 = obj.view(_np.ndarray)
    obj1[sl_in] = item
    obj1 = obj1.view(type(obj))
    obj1.__dict__.update(obj.__dict__)
    obj = obj1


class scatteringMatrix(gpri_files.gammaDataset):
    pauli_basis = [_np.array([[1, 0], [0, 1]]) * 1 / _np.sqrt(2), _np.array([[1, 0], [0, -1]]) * 1 / _np.sqrt(2),
                   _np.array([[0, 1], [1, 0]]) * 1 / _np.sqrt(2)]
    lexicographic_basis = [_np.array([[1, 0], [0, 0]]) * 2, _np.array([[0, 1], [0, 0]]) * 2 * _np.sqrt(2),
                           _np.array([[0, 0], [0, 1]]) * 2]

    def __new__(cls, *args, gpri=False, suffix='slc', H_ant='A', memmap=False, sl=[Ellipsis] * 2, chan='l'):
        """
        This function loads data saved as scattering matrix from a file or initialised it from a 3X3 / nXmX3X3 array

        Parameters
        ----------
        gpri : bool
            If set to true, loads a gpri dataset. In this case, the first positional argument is a string giving the basepath of the dataset
        chan : string
            String to specifiy wheter to use the upper or the lower GPRI channel
        suffix : string
            The file suffix for the gamma dataset to load.
        H_ant : string
            Either `A` or `B`, depending on the channel mapping between GPRI RF port and polarimetric antennas

        Returns
        -------
        scatteringMatrix
            A scattering matrix object loaded from the specified path. The attributes are taken from the HH channel fileutils.gpri_files.gammaDataset
        """
        if gpri:
            chan = chan
            sl = sl
            memmap = memmap
            H_ant = H_ant
            V_ant = 'B' if H_ant == 'A' else 'A'
            suffix = suffix
            base_path = args[0]
            # Used to index the matrix
            lst_tx = [0, 1]
            lst_rx = [0, 1]
            # load HH parameters to obtain info on shape etc
            HH_par = gpri_files.par_to_dict(base_path + "_" + "AAAl." + suffix + ".par")
            shpe = (HH_par['range_samples'], HH_par['azimuth_lines'])
            dt = gpri_files.type_mapping[HH_par['image_format']]
            # Create memmaps if the output is to be stored in such a format
            if memmap:
                mat_path = base_path + 's_matrix_' + chan
                open(mat_path, 'w+').close()
                s_matrix = _np.memmap(mat_path,
                                      dtype=dt, shape=shpe + (2, 2),
                                      mode='r+')
            else:
                s_matrix = _np.zeros(shpe + (2, 2), dtype=dt)
            s_matrix = _np.zeros(shpe + (2, 2), dtype=dt)
            # Containers for parameters and phase centers for each channel
            phase_center_array = {}
            par_array = {}
            for tx, idx_tx, tx_name in zip([H_ant, V_ant], lst_tx, ['H', 'V']):
                for rx, idx_rx, rx_name in zip([H_ant, V_ant], lst_rx, ['H', 'V']):
                    file_pattern = base_path + "_" + tx + rx + rx + chan + '.' + suffix
                    ds = gpri_files.gammaDataset(file_pattern + ".par", file_pattern, memmap=memmap)
                    s_matrix[:, :, idx_tx, idx_rx] = ds
                    # Set phase center
                    phase_center_array[tx_name + rx_name] = ds.phase_center
                    par_array[tx_name + rx_name] = ds.__dict__

            if memmap:
                s_matrix.flush()
            obj = s_matrix.view(cls)
            # Add attributes
            # Copy attributes from one channel
            obj.__dict__ = _cp.deepcopy(HH_par)
            obj.geometry = 'polar'
            obj.phase_center_array = phase_center_array
            obj.par_array = par_array
        else:
            if isinstance(args[1], _np.ndarray):
                s_matrix = args[1]
            elif isinstance(args[1], str):
                args = args[1:None]
                s_matrix = other_files.load_scattering(*args, **kwargs)

            obj = s_matrix.view(scatteringMatrix)
            obj.geometry = 'cartesian'
        return obj

    def from_gpri_to_normal(self):
        self_1 = self[:]
        self_1.geometry = None
        return self_1

    def scattering_vector(self, basis='pauli', bistatic=True):
        """ 
        This function returns the scattering vector in the pauli or lexicographics basis
    
        Parameters
        ----------
        basis : string
            The basis of interest. Can be either 'pauli' or 'lexicographic'
        bistatic :  bool
            Set to True if the bistatic scattering vector is needed
        Returns
        -------
        ndarray
            The resulting scattering vector
        """
        # Generate necessary transformation matrices
        if bistatic is True:
            sv = _np.zeros(self.shape[0:2] + (4,), dtype=self.dtype)
            if basis is 'pauli':
                sv1 = _np.array(self['HH'] + self['VV'])
                sv2 = _np.array(self['HH'] - self['VV'])
                sv3 = _np.array(self['HV'] + self['VH'])
                sv4 = _np.array(1j * (self['HV'] - self['VH']))
                factor = 1 / _np.sqrt(2)
            elif basis is 'lexicographic':
                sv1 = _np.array(self['HH'])
                sv2 = _np.array(self['HV'])
                sv3 = _np.array(self['VH'])
                sv4 = _np.array(self['VV'])
                factor = 1
            if self.ndim is 2:
                sv = factor * _np.hstack((sv1, sv2, sv3, sv4))
            else:
                sv = factor * _np.dstack((sv1, sv2, sv3, sv4))
        elif bistatic is False:
            sv = _np.zeros(self.shape[0:2] + (3,), dtype=self.dtype)
            if basis is 'pauli':
                sv1 = _np.array(self['HH'] + self['VV'])
                sv2 = _np.array(self['HH'] - self['VV'])
                sv3 = _np.array((self['HV']) * 2)
                factor = 1 / _np.sqrt(2)
            elif basis is 'lexicographic':
                sv1 = _np.array(self['HH'])
                sv2 = _np.array(_np.sqrt(2) * self['HV'])
                sv3 = _np.array(self['VV'])
                factor = 1
            if self.ndim is 2:
                sv = factor * _np.hstack((sv1, sv2, sv3))
            else:
                sv = factor * _np.dstack((sv1, sv2, sv3))
        return sv

    def span(self):
        """ 
        This function computes the polarimetric span of the scattering matrix
        Parameters
        ----------

        Returns
        -------
        sp: ndarray
            The resulting span
        """
        v = self.scattering_vector(basis='lexicographic', bistatic=True)
        if self.ndim is 4:
            ax = 2
        else:
            ax = 0
        sp = _np.sum(_np.abs(v) ** 2, axis=ax)
        return sp

    def symmetrize(self):
        """ 
        This function symmetrizes the scattering matrix
        """
        Z_hv = (self['HV'] + (self['VH'] * _np.exp(-1j * (_np.angle(self['HV']) - _np.angle(self['VH']))))) / 2
        self['HV'] = Z_hv
        self['VH'] = Z_hv

    def to_coherency_matrix(self, bistatic=False, basis='pauli', ):
        """ 
        This function converst the scattering matrix into a coherency matrix
        using the chosen basis. It does not perform any averaging, so the resulting
        matrix has rank 1.
        
        Parameters
        ----------

        bistatic : bool
            If set to true, the bistatci coherency matric is computed
        basis : string
            Allows to choose the basis for the scattering vector

        Returns
        -------

        coherencyMatrix
            the resulting coherency matrix
        """
        k = self.scattering_vector(bistatic=bistatic, basis=basis)
        T = corefun.outer_product(k, k)
        T = T.view(coherencyMatrix)
        T.__dict__.update(self.__dict__)
        T.__dict__['basis'] = basis
        return T

    def pauli_image(self, **kwargs):
        return gpi(self, **kwargs)


#




class coherencyMatrix(gpri_files.gammaDataset):
    U3LP = 1 / _np.sqrt(2) * _np.array([[1, 0, 1], [1, 0, -1], [0, _np.sqrt(2), 0]])
    U4LP = 1 / _np.sqrt(2) * _np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]])
    U4PL = U4LP.T.conj()
    U3PL = U3LP.T.conj()

    def vectorize(self):
        dim = self.shape
        if dim[-1] is 3:
            access_vector = [0, 4, 8, 1, 2, 5, 3, 6, 7]
        elif dim[-1] is 4:
            access_vector = [0, 5, 10, 15, 1, 2, 3, 6, 7, 11, 4, 8, 9, 12, 13, 14]
        else:
            access_vector = range(dim[-1])
        if self.ndim is 4:
            new_self = _np.array(self).reshape(self.shape[0:2] + (self.shape[-1] ** 2,))[:, :, access_vector]
        else:
            new_self = _np.array(self).reshape((self.shape[-1] ** 2,))[access_vector]
        return new_self

    def normalize(self):
        span = self.span()
        if self.ndim == 2:
            T_norm = self / span
        elif self.ndim == 4:
            T_norm = self / span[:, :, None, None]
        elif self.ndim == 3:
            T_norm = self / span[:, None, None]
        return T_norm

    def cloude_pottier(self):
        l, w = _np.linalg.eigh(self)
        l = _np.array(l)
        w = _np.array(w)
        if self.ndim is 4:
            l_sum = _np.sum(_np.abs(l), axis=2)
            p = l / l_sum[:, :, _np.newaxis]
            H = -_np.sum(p * _np.log10(p) / _np.log10(3), axis=2)
            alpha = _np.arccos(_np.abs(w[:, :, 0, 0:None]))
            rotated = w[:, :, 1, 0:None] * 1 / _np.sin(alpha)
            beta = _np.arccos(_np.abs(rotated))
            anisotropy = (l[:, :, 1] - l[:, :, 2]) / (l[:, :, 1] + l[:, :, 2])
            alpha_m = _np.sum(alpha * p, axis=2)
            beta_m = _np.sum(beta * p, axis=2)
        if self.ndim is 2:
            l_sum = _np.sum(l, axis=1)
            p = l / l_sum[:, _np.newaxis]
            H = -_np.sum(p * _np.log10(p) / _np.log10(3), axis=1)
            alpha = _np.arccos(_np.abs(w[:, 0, 0:None]))
            alpha_m = _np.sum(alpha * p, axis=1)
            anisotropy = (l[1] - l[2]) / (l[1] + l[2])
        return H, anisotropy, alpha_m, beta_m, p, w

    def transform(self, T1, T2):
        out = self.__array_wrap__(_np.einsum("...ik,...kl,...lj->...ij", T1, self, T2))
        return out

    def rank_image(self, threshold):
        image = self
        l, u = _np.linalg.eigh(image)
        dist = lambda x, y: _np.abs(x - y) < threshold
        c1 = dist(l[:, :, 0], l[:, :, 1])
        c2 = dist(l[:, :, 0], l[:, :, 2])
        c3 = dist(l[:, :, 1], l[:, :, 2])
        counts = c1.astype(_np.int) + c2.astype(_np.int) + c3.astype(_np.int)
        return counts

    def to_monostatic(self):
        self.bistatic = False
        i, j = _np.meshgrid([0, 1, 3], [0, 1, 3])
        if self.ndim is 4:
            return self[:, :, i, j]
        else:
            return self[i, j]

    def __new__(*args, **kwargs):
        cls = args[0]
        # The type of data
        basis = kwargs.get('basis', 'lexicographic')
        # bistatic = kwargs.get('bistatic')
        # Get keywords
        coherency = kwargs.get('coherency')
        agrisar = kwargs.get('agrisar')
        polsarpro = kwargs.get('polsarpro')
        gamma = kwargs.get('gamma')
        dim = kwargs.get('dim')
        bistatic = kwargs.get('bistatic', True)
        suffix = kwargs.get('suffix', '.c{i}{j}')
        if type(args[1]) is _np.ndarray:  # from array
            T = args[1]
            # TODO check why strange hermitian behavior for GPRI data
            obj = _np.asarray(T).view(cls)
            obj.basis = basis
        elif type(args[1]) is str:
            path = args[1]
            if agrisar:
                s_matrix = scatteringMatrix(path, fmt='esar')
                pauli = s_matrix.scattering_vector(bistatic=bistatic, basis=basis)
                T = corefun.outer_product(pauli, pauli)
            elif coherency:
                T = other_files.load_coherency(path, dim)
            elif polsarpro:
                s_matrix = scatteringMatrix(path, dim, fmt='polsarpro')
                # Pauli representation is defaulto
                pauli = s_matrix.scattering_vector(bistatic=bistatic, basis=basis)
                T = corefun.outer_product(pauli, pauli)
            elif gamma:
                basis = 'lexicographic'
                par_name = args[2]
                if bistatic:
                    chan_dict = {0: 0, 1: 1, 2: 2, 3: 3}
                else:
                    chan_dict = {0: 1, 1: 1, 2: 3}
                # Load shape
                par_dict = gpri_files.par_to_dict(par_name)
                shp = (par_dict['range_samples'], par_dict['azimuth_lines'])
                C = _np.zeros(shp + (len(chan_dict), len(chan_dict)), dtype=_np.complex64)
                for chan_1 in chan_dict.keys():
                    for chan_2 in chan_dict.keys():
                        extension = suffix.format(i=chan_dict[chan_1], j=chan_dict[chan_2])
                        chan_name = path + extension
                        chan, par = gpri_files.load_dataset(par_name, chan_name)
                        C[:, :, chan_1, chan_2] = chan
                T = C
        # Finally, we must return the newly created object:
        obj = T.view(cls)
        obj.window = [1, 1]
        obj.basis = basis
        obj.geometry = 'cartesian'
        if gamma:
            obj.__dict__.update(par)
        return obj

    def boxcar_filter(self, window, discard=False):
        """
        This function applies boxcar averaging on the coherency matrix
        Parameters
        ----------
        window : tuple
            a tuple of window sizes
        discard :  bool, optional
            set to true if only the centra pixel of the window has to be kept
        """
        T = self.__array_wrap__(corefun.smooth(self, window + [1, 1]))
        if discard:
            T = T[0:None:window[0], 1:None:window[1], :, :]
        return T

    def span(self):
        shp = self.shape
        if len(shp) is 4:
            s = _np.trace(self, axis1=2, axis2=3)
        else:
            s = _np.trace(self)
        s1 = _np.real(s.view(_np.ndarray))

        return s1

    def pauli_image(self, **kwargs):
        return gpi(self, **kwargs)

    def generateRealizations(self, n_real, n_looks):
        #        #Generate unit vectors
        n_tot = n_real * n_looks
        k = _np.random.multivariate_normal(_np.zeros(3), self, n_tot)
        k = k.transpose()
        outers = _np.einsum('i...,j...', k, k.conj())
        outers = _np.reshape(outers, (n_real, n_looks, 3, 3))
        outers = _np.mean(outers, axis=1)
        return outers.view(coherencyMatrix)

    def pauli_to_lexicographic(self):
        """
        This function converst the current matrix to the lexicographic basis


        """
        if self.basis == 'pauli':
            if self.shape[-1] is 3:
                C = self.transform(self.U3PL, self.U3LP)
            else:
                C = self.transform(self.U4PL, self.U4LP)
            C.basis = 'lexicographic'
        else:
            C = self
        C = self.__array_wrap__(C)
        return C

    def lexicographic_to_pauli(self):
        """
        This function converst the current matrix to the pauli basis
        """
        if self.basis == 'lexicographic':
            if self.shape[-1] is 3:
                C = self.transform(self.U3LP, self.U3PL)
            else:
                C = self.transform(self.U4LP, self.U4PL)
            C.basis = 'pauli'
        else:
            C = self
        C = self.__array_wrap__(C)
        return C

    def to_gamma(*args, **kwargs):
        bistatic = kwargs.get('bistatic', False)
        self = args[0]
        root_name = args[1]
        if self.basis is 'lexicographic':
            ending = 'c'
        else:
            ending = 't'
        if bistatic:
            chan_dict = {0: 0, 1: 1, 2: 2, 3: 3}
        else:
            chan_dict = {0: 0, 1: 1, 2: 3}
        for chan_1 in chan_dict.keys():
            for chan_2 in chan_dict.keys():
                extension = ".c{i}{j}".format(i=chan_dict[chan_1], j=chan_dict[chan_2])
                chan_name = root_name + extension
                _np.array(self[:, :, chan_1, chan_2]).T.astype(gpri_files.type_mapping['FCOMPLEX']).tofile(chan_name)
        gpri_files.dict_to_par(self.__dict__, root_name + '.par')


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h // nrows, -1, nrows, ncols)
            .swapaxes(1, 2)
            .reshape(h, w))


# TODO fix block processing
class block_array:
    def __init__(*args):

        obj = args[0]
        # The array to be split
        A = args[1]
        # Size of desired blocks
        block_size = args[2]
        # Size of the processing window
        window_size = args[3]
        # 2D shape of array
        shape_2d = A.shape[0:2]
        # Create object
        obj.bs = block_size
        obj.A = A
        # iterate over block sizes to compute indices for each dim
        obj.rsa = []
        obj.rea = []
        obj.wsa = []
        obj.wea = []
        # Current index for the iterator
        obj.current = 0
        obj.nblocks = []
        for bs, wins, ars in zip(block_size, window_size, shape_2d):
            # Compute number of block
            N_block = _np.ceil((ars - wins + 1) / (bs - wins + 1))
            block_idx = _np.arange(N_block)
            rs = block_idx * (bs - wins + 1)
            re = rs + bs - 1
            re[-1] = ars - 1
            ws = _np.zeros(N_block) + (wins - 1) / 2
            we = _np.zeros(N_block) + bs - (wins + 1) / 2
            ws[0] = 0
            re[-1] = ars - 1
            we[-1] = ars - 1 - rs[-1]
            obj.rsa.append(rs)
            obj.rea.append(re)
            obj.wsa.append(ws)
            obj.wea.append(we)
            obj.nblocks.append(N_block)
        obj.maxiter = _np.prod(obj.nblocks)

    def __getitem__(self, sl):
        if len(sl) is 1:
            return self.take_block(sl)

    def __setitem__(self, sl, item):
        self.put_block(sl, item)

    def __iter__(self):
        return self

    def next(self):
        if self.current >= self.maxiter:
            self.current = 0
            raise StopIteration
        else:
            c = self.take_block(self.current)
            self.current += 1
            return c

    def put_current(self, item):
        self.put_block(self.current, item)

    def center_index(self, idx):
        try:
            return _np.unravel_index(idx, self.nblocks)
        except:
            return 0, 0

    def take_block(self, idx):
        if idx < _np.prod(self.nblocks):
            i, j = _np.unravel_index(idx, self.nblocks)
            block = self.A[self.rsa[0][i]:self.rsa[0][i] + self.rea[0][i],
                    self.rsa[1][j]:self.rsa[1][j] + self.rea[1][j]]
        return block

    def put_block(self, idx, block):

        if idx < _np.prod(self.nblocks):
            i, j = _np.unravel_index(idx, self.nblocks)
            clipped_block = block[self.wsa[0][i]:self.wsa[0][i] + self.wea[0][i],
                            self.wsa[1][j]:self.wea[1][j] + self.wsa[1][j]]
            start_i = self.rsa[0][i] + self.wsa[0][i]
            start_j = self.rsa[1][j] + self.wsa[1][j]
            self.A[start_i:start_i + clipped_block.shape[0],
            start_j:start_j + clipped_block.shape[1]] = clipped_block

# -*- coding: utf-8 -*-
"""
Created on Thu May 15 13:51:30 2014

This module contains classes and function to deal with Gamma file formats


@author: baffelli
"""
import copy as _cp
import mmap as _mm
import os as _os
import os.path as _osp
import re as _re
import sys as _sys
import tempfile as _tf
from collections import OrderedDict as _od
from collections import namedtuple as _nt

import numpy as _np
import scipy as _sp
import scipy.signal as _sig
from numpy.lib.stride_tricks import as_strided as _ast

# Constants for gpri
ra = 6378137.0000  # WGS-84 semi-major axis
rb = 6356752.3141  # WGS-84 semi-minor axis
C = 299792458.0  # speed of light m/s
# 15.7988 x 7.8994 mm
KU_WIDTH = 15.798e-3  # WG-62 Ku-Band waveguide width dimension
KU_DZ = 10.682e-3  # Ku-Band Waveguide slot spacing
KU_DZ_ALT = 10.3e-3  # Ku-Band Waveguide slot spacing
RANGE_OFFSET = 3
xoff = 0.112  # 140mm X offset to the antenna holder rotation axis
ant_radius = 0.1115  # 99.2mm radial arm length. This is the rotation radius of the antenna holder
# Z coordinates of antennas W.R.T the bottommost antenna, up is +Z
tx_dz = {'A': 0.85, 'B': 0.725}
rx1_dz = {'A': 0.375, 'B': 0.25}
rx2_dz = {'A': 0.125, 'B': 0}
# Scaling factor short integer <-> float
TSF = 32768

# This dict defines the mapping between
# upper and lower channels and indices used to access them
# raw file are interleaved, ch1, ch2, ch1, ch2
ant_map = {'l': 0, 'u': 1}

# This dict defines the mapping
# between the gamma datasets and numpy
type_mapping = {
    'FCOMPLEX': _np.dtype('>c8'),
    # 'SCOMPLEX': _np.dtype('>c2'),
    'FLOAT': _np.dtype('>f4'),
    'SHORT INTEGER': _np.dtype('<i2'),
    'INTEGER*2': _np.dtype('>i2'),
    'INTEGER': _np.dtype('>i'),
    'REAL*4': _np.dtype('>f4'),
    'UCHAR': _np.dtype('>b')
}

ls_mapping = {
    0: 'NOT_TESTED',
    1: 'TESTED',
    2: 'TRUE_LAYOVER',
    4: 'LAYOVER',
    8: 'TRUE_SHADOW',
    16: 'SHADOW'
}

# This dict defines the mapping
# between channels in the file name and the matrix
channel_dict = {
    'HH': (0, 0),
    'HV': (0, 1),
    'VH': (1, 0),
    'VV': (1, 1),
}


def gamma_datatype_code_from_extension(filename):
    """
        Get the numeric datatype from the filename extensions
    """
    mapping = {'slc': 1,
               'slc_dec': 1,
               'mli': 0,
               'mli_dec': 0,
               'inc': 0,
               'dem_seg': 0,
               'dem': 0,
               'int': 1,
               'sm': 1,
               'cc': 0,
               "sim_sar": 0,
               "ls_map": 3,
               "sh_map": 3,
               "u": 0,
               "lv_theta": 0,
               "bmp": 2}
    for i in [0, 1, 2, 3]:
        for j in [0, 1, 2, 3]:
            mapping["c{i}{j}".format(i=i, j=j)] = 1  # covariance elements
            # Split the file name to the latest extesnsion
    extension = filename.split('.')[-1]
    return mapping[extension]


def gt_mapping_from_extension(filename):
    """
        Get numeric datatype from file extension for
        conversion into geotiff
    """
    mapping = {
        'slc': 4,
        'mli_dec': 2,
        'bmp': 0,
        'sim_sar': 2,
        'ls_map': 5,
        'dem_seg': 4,
        'u': 2,
        'lv_theta': 2,
        'sh_map': 5,
    }
    extension = filename.split('.')[-1]
    extension = _re.sub("(_(f)*(gc))+", "", extension)
    print(filename)
    return mapping[extension]


ls_map_dic = {0: "NOT_TESTED",
              1: "TESTED",
              2: "TRUE_LAYOVER",
              4: "LAYOVER",
              8: 'TRUE_SHADOW',
              16: 'SHADOW'
              }


def get_image_size(path, width, type_name):
    """
    This function gets the shape of an image given the witdh
    and the datatype
    :param path:
    :param width:
    :param type:
    :return:
    """
    fc = _os.path.getsize(path) / type_mapping[type_name].itemsize
    shape = [width, int(fc / width)]
    computed_size = shape[0] * shape[1] * type_mapping[type_name].itemsize
    measured_size = _os.path.getsize(path)
    return shape


def temp_dataset():
    """
    This function produces a temporary dataset which uses a temporary
    file for both the binary  file and the parameter file
    :return:
    two file descriptors. the first for the parameter and the second for
    the binary file
    """
    tf_par = _tf.NamedTemporaryFile(delete=False, mode='w+t')
    tf = _tf.NamedTemporaryFile(delete=False)
    return tf_par, tf


def temp_binary(suffix=''):
    return _tf.NamedTemporaryFile(delete=False, suffix=suffix)


def to_bitmap(dataset, filename):
    fmt = 'bmp'
    _sp.misc.imsave(filename, dataset, format=fmt)


def load_plist(path):
    """
    Loads a point list from a gamma file
    Parameters
    ----------
    path
        the path of the point list
    Returns
    -------
        the point list as a 2d array
    """
    dt = _np.dtype({"names": ['ridx', 'azidx'],
                    "formats": ['>u4', '>u4']})
    plist = _np.fromfile(path, dtype=dt)
    return plist


class gammaDataset(_np.ndarray):
    def __new__(cls, *args, **kwargs):
        par_dict = args[0]
        image = args[1]
        if isinstance(par_dict, str):#user passes paths
            print('hier')
            try:#user passes file paths
                par_path = args[0]
                bin_path = args[1]
                memmap = kwargs.get('memmap', False)
                dtype = kwargs.get('dtype', None)
                image, par_dict = load_dataset(par_path, bin_path, memmap=memmap, dtype=dtype)
            except:
                Exception("Input parameters format unrecognized ")
        else:#user passes binary and dictionary
            pass
        obj = image.view(cls)
        d1 = _cp.deepcopy(par_dict)
        obj.__dict__ = d1
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        if hasattr(obj, '__dict__'):
            self.__dict__ = _cp.deepcopy(obj.__dict__)

    def __getslice__(self, start, stop):
        """This solves a subtle bug, where __getitem__ is not called, and all
        the dimensional checking not done, when a slice of only the first
        dimension is taken, e.g. a[1:3]. From the Python docs:
           Deprecated since version 2.0: Support slice objects as parameters
           to the __getitem__() method. (However, built-in types in CPython
           currently still implement __getslice__(). Therefore, you have to
           override it in derived classes when implementing slicing.)
        """
        return self.__getitem__(slice(start, stop, None))

    def __getitem__(self, item):
        # Try the channel dictionary
        try:
            sl_mat = channel_dict[item]
            sl = (Ellipsis,) * (self.ndim - len(sl_mat)) + sl_mat
        except (KeyError, TypeError):
            sl = item
        # TODO Fix spacing
        # Get the slice from the object by calling the corresponding numpy function
        new_obj_1 = (super(gammaDataset, self).__getitem__(sl)).view(type(self))
        # #This concludes the part where we extract data from the array.
        # #now we need to adjust the attributes to adjust to the new spacing
        # try:
        #     az_vec = self.az_vec * 1
        #     r_vec = self.r_vec * 1
        #     #copy range and azimuth vector
        #     #Now access the range vector
        #     try:
        #         r_vec_sl = r_vec[sl[0]]
        #     except (IndexError, TypeError):
        #         try:
        #             r_vec_sl = r_vec[sl]
        #         except IndexError:
        #             r_vec_slc = r_vec
        #     try:
        #         az_vec_sl = az_vec[sl[1]]
        #     except (IndexError, TypeError):
        #         try:
        #             az_vec_sl = az_vec[sl]
        #         except IndexError:
        #             az_vec_sl = az_vec
        #
        #     try:
        #         az_osf = (az_vec_sl[1] -  az_vec_sl[0])/ self.GPRI_az_angle_step[0]#azimuth over/undersampling time
        #     except IndexError:
        #         az_osf = 1
        #     try:
        #         r_osf = (r_vec_sl[1] -  r_vec_sl[0])/ self.range_pixel_spacing[0]#range sampling
        #     except IndexError:
        #         r_osf = 1
        #     new_obj_1.azimuth_line_time[0] = az_osf * self.azimuth_line_time[0]
        #     new_obj_1.prf[0] = az_osf * self.prf[0]
        #     try:
        #         start_angle = az_vec_sl[0]
        #     except IndexError:
        #         start_angle = az_vec_sl#the azimuth vector is a single number ("object sliced to death")
        #     try:
        #         start_r = az_vec_sl[0]
        #     except IndexError:
        #         start_r = az_vec_sl
        #     try:
        #         new_obj_1.near_range_slc[0] = start_r
        #         new_obj_1.GPRI_az_start_angle[0] = start_angle
        #         new_obj_1.GPRI_az_angle_step[0] = self.GPRI_az_angle_step[0] * az_osf
        #         new_obj_1.range_pixel_spacing[0] = self.range_pixel_spacing[0] * r_osf
        #         new_obj_1.azimuth_line_time[0] = az_osf * self.azimuth_line_time[0]
        #         new_obj_1.prf[0] = az_osf * self.prf[0]
        #         new_obj_1.range_samples = new_obj_1.shape[0]
        #         new_obj_1.azimuth_lines = new_obj_1.shape[1] if new_obj_1.ndim > 1 else 0
        #     except AttributeError:
        #         pass
        # except (AttributeError, KeyError):
        #     pass#not a gpri dataset

        # # Construct temporary azimuth and  range vectors
        # az_vec = self.az_vec * 1
        # r_vec = self.r_vec * 1
        # r_0 = self.r_vec[0]
        # az_0 = self.az_vec[0]
        # az_spac = self.GPRI_az_angle_step[0] * 1
        # r_spac = self.range_pixel_spacing[0] * 1
        # # Passing only number, access the array as it were a linear index (ravel index)
        # if isinstance(sl, _num.Number):
        #     #compute the indices and recur
        #     indices = _np.unravel_index(sl)
        #     return self.__getitem__(indices)
        #
        # # Tuple of slices
        # elif hasattr(sl, '__contains__'):
        #     # By taking the first element, we automatically have
        #     # the correct data
        #     try:
        #         az_vec_sl = self.az_vec[sl[1]]
        #         if hasattr(az_vec_sl, '__contains__'):
        #             if len(az_vec_sl) > 1:
        #                 az_spac = az_vec_sl[1] - az_vec_sl[0]
        #                 az_0 = az_vec_sl[0]
        #             else:
        #                 az_spac = self.GPRI_az_angle_step[0] * 1
        #                 az_0 = az_vec_sl
        #         else:
        #             az_0 = az_vec_sl
        #             az_spac = self.GPRI_az_angle_step[0] * 1
        #     except:
        #         IndexError('The slice object does not match the objects size')
        #     try:
        #         r_vec_sl = self.r_vec[sl[0]]
        #         if hasattr(r_vec_sl, '__contains__'):
        #             if len(r_vec_sl) > 1:
        #                 r_spac = r_vec_sl[1] - r_vec_sl[0]
        #             else:
        #                 r_spac = self.range_pixel_spacing[0]
        #             r_spac = r_vec_sl[1] - r_vec_sl[0]
        #             r_0 = r_vec_sl[0]
        #         else:
        #             r_spac = self.range_pixel_spacing[0] * 1
        #             r_0 = r_vec_sl
        #     except:
        #         pass
        # az_osf = az_spac / self.GPRI_az_angle_step[0]#azimuth over/undersampling time
        # new_obj_1.azimuth_line_time[0] = az_osf * self.azimuth_line_time[0]
        # new_obj_1.prf[0] = az_osf * self.prf[0]
        # new_obj_1.GPRI_az_start_angle[0] = az_0
        # new_obj_1.near_range_slc[0] = r_0
        # new_obj_1.GPRI_az_angle_step[0] = az_spac
        # new_obj_1.range_pixel_spacing[0] = r_spac
        # new_obj_1.range_samples = new_obj_1.shape[0]
        # new_obj_1.azimuth_lines = new_obj_1.shape[1] if new_obj_1.ndim > 1 else 0
        return new_obj_1

    def tofile(*args):
        self = args[0].astype(type_mapping[args[0].image_format])
        # In this case, we want to write both parameters and binary file
        if len(args) is 3:
            write_dataset(self, self.__dict__, args[1], args[2])
        # In this case, we only want to write the binary
        else:
            _np.array(self).tofile(args[1])

    def __setitem__(self, key, value):
        # Construct indices
        if type(key) is str:
            try:
                sl_mat = channel_dict[key]
                sl = (Ellipsis,) * (self.ndim - 2) + sl_mat
            except KeyError:
                raise IndexError('This channel does not exist')
        else:
            sl = key
        super(gammaDataset, self).__setitem__(key, value)

    def to_tempfile(self):
        tf_par, tf = temp_dataset()
        self.tofile(tf_par.name, tf.name)
        return tf_par, tf

    def decimate(self, dec):
        arr_dec = _np.zeros((self.shape[0], int(self.shape[1] / dec)), dtype=_np.complex64)
        for idx_az in range(arr_dec.shape[1]):
            # Decimated pulse
            dec_pulse = _np.zeros(self.slc.shape[0] * 2 - 2, dtype=_np.float32)
            for idx_dec in range(self.dec):
                current_idx = idx_az * self.dec + idx_dec
                if current_idx % 1000 == 0:
                    print('decimating line: ' + str(current_idx))
                dec_pulse += _np.fft.irfft(self.slc[:, current_idx])
            arr_dec[:, idx_az] = _np.fft.rfft(dec_pulse)
        return arr_dec

    @property
    def r_vec(self):
        return self.__dict__['near_range_slc'][0] + _np.arange(self.__dict__['range_samples']) * \
                                                    self.__dict__['range_pixel_spacing'][0]

    @property
    def az_vec(self):
        return self.__dict__['GPRI_az_start_angle'][0] + _np.arange(self.__dict__['azimuth_lines']) * \
                                                         self.__dict__['GPRI_az_angle_step'][0]

    @property
    def phase_center(self):

        """
        This function computes the phase center position for a given slc
        file by computing (rx_pos + tx_pos)/2
        Parameters
        ----------
        par, dict
        Dictionary of slc parameters in the gamma format

        Returns
        -------

        """
        try:
            rx_number = extract_channel_number(self.title[-1])
            ph_center = (_np.array(self.GPRI_tx_coord[0:3]) + _np.array(
                getattr(self, "GPRI_rx{num}_coord".format(num=rx_number))[0:3])) / 2
            return ph_center
        except AttributeError:
            return 0


def par_to_dict(par_file):
    """
    This function converts a gamma '.par' file into
    a dict of parameters
    :param par_file:
    A string containing the path to the parameter file
    :return:
    A dict of parameters
    """
    par_dict = _od()
    with open(par_file, 'r') as fin:
        # Skip first line
        # fin.readline()
        for line in fin:
            if line:
                split_array = line.replace('\n', '').split(':', 1)
                if len(split_array) > 1:
                    key = split_array[0]
                    if key == 'time_start':  # The utc time string should not be split
                        l = split_array[1:]
                    else:
                        l = []
                        param = split_array[1].split()
                        for p in param:
                            try:
                                l.append(float(p))
                            except ValueError:
                                l.append(p)
                try:
                    if len(l) > 1:
                        par_dict[key] = l
                    else:
                        par_dict[key] = l[0]
                except:
                    pass
    return par_dict


def get_width(par_path):
    """
        Helper function to get the width of a gamma format file
    """
    par_dict = par_to_dict(par_path)
    for name_string in ["width", "range_samples", "CHP_num_samp", "number_of_nonzero_range_pixels_1",
                        "interferogram_width"]:
        try:
            width = par_dict[name_string]
        except:
            pass
        else:
            break
    return width


def datatype_from_extension(filename):
    """
        Get the numeric datatype from the filename extensions
    """
    mapping = {'slc': 1,
               'slc_dec': 1,
               'mli': 0,
               'mli_dec': 0,
               'inc': 0,
               'dem_seg': 0,
               'dem': 0,
               'int': 1,
               'sm': 1,
               'cc': 0,
               "sim_sar": 0,
               "ls_map": 3,
               "bmp": 2}
    for i in [0, 1, 2, 3]:
        for j in [0, 1, 2, 3]:
            mapping["c{i}{j}".format(i=i, j=j)] = 1  # covariance elements
            # Split the file name to the latest extesnsion
    extension = filename.split('.')[-1]
    return mapping[extension]


def dict_to_par(par_dict, par_file):
    """
    This function writes a dict to a gamma
    format parameter file
    :param par_dict:
    A dict of parameters
    :param par_file:
    A string with the path to the parameter file
    :return:
    None
    """
    with open(par_file, 'w') as fout:
        for key in iter(par_dict):
            par = par_dict[key]
            out = str(key) + ":" + '\t'
            if isinstance(par, str):
                out += par
            else:
                try:
                    for p in par:
                        out = out + ' ' + str(p)
                except TypeError:
                    out += str(par)
            fout.write(out + '\n')


def load_binary(bin_file, width, dtype=type_mapping['FCOMPLEX'], memmap=False):
    # Get filesize
    filesize = _osp.getsize(bin_file)
    # Get itemsize
    itemsize = dtype.itemsize
    # Compute the number of lines
    nlines = filesize / (itemsize * width)
    # Shape of binary
    shape = (width, nlines)
    # load binary
    if memmap:
        with open(bin_file, 'rb') as mmp:
            buffer = _mm.mmap(mmp.fileno(), 0, prot=_mm.PROT_READ)
            d_image = _np.ndarray(shape[::-1], dtype, buffer).T
    else:
        d_image = _np.fromfile(bin_file, dtype=dtype).reshape(shape[::-1]).T
    return d_image


def load_dataset(par_file, bin_file, **kwargs):
    dtype = kwargs.get('dtype', None)
    memmap = kwargs.get('memmap', False)
    par_dict = par_to_dict(par_file)
    # Map type to gamma
    if dtype is None:
        try:
            dt = type_mapping[par_dict['image_format']]
        except:
            try:
                dt = type_mapping[par_dict['data_format']]
            except:
                dt = type_mapping['FLOAT']
                print(str(KeyError(
                    "This file does not contain datatype specification in a known format, using default FLOAT datatype")))
    else:
        try:
            dt = dtype
        except KeyError:
            raise TypeError('This datatype does not exist')
    try:
        width = par_dict['range_samples']
    except:
        # We dont have a SAR image,
        # try as it were a DEM
        try:
            width = par_dict['nlines']
        except:
            # Last try
            # interferogram
            try:
                width = par_dict['interferogram_width']
            except:
                try:
                    width = par_dict['range_samp_1']
                except:
                    raise KeyError("This file does not contain data shape specification in a known format")
    d_image = load_binary(bin_file, width, dtype=dt, memmap=memmap)
    return d_image, par_dict


def write_dataset(dataset, par_dict, par_file, bin_file):
    # Write the  binary file
    try:
        _np.array(dataset).T.tofile(bin_file, "")
    except AttributeError:
        raise TypeError("The dataset is not a numpy ndarray")
    # Write the parameter file
    dict_to_par(par_dict, par_file)


def path_helper(location, date, time, slc_dir='slc', data_dir='/media/bup/Data'):
    """
    This is an helper function to construct the paths
    to load slc file in the usual order they are stored
    i.e '/location/date/slc/date_time.slc'
    Parameters
    ----------
        location : str
            The location of acquistion
        date : str
            Date
        time : str
            Time of the acquistion
        slc_dir : str
            name of the slc subdirectory
        data_dir : str
            absolute path to where the data is stored
    """

    base_folder = data_dir + '/' + location + '/' + date + '/'
    name = date + '_' + time
    def_path = base_folder + slc_dir + '/' + name
    return def_path


def extract_channel_number(title):
    """
    This function extract the channel numer (lower 1 or upper 2) from the title
    of the slc file, which in in the format
    "title:	 2015-12-07 14:20:27.671486+00:00 CH2 upper"

    Parameters
    ----------
    title

    Returns
    -------

    """
    # Generate re
    p = _re.compile("(lower)|(upper)")
    result = _re.search(p, title)
    idx = result.lastindex
    return idx


def gpri_raw_strides(nsamp, nchan, npat, itemsize):
    """
    This function computes the array strides for 
    the gpri raw file format
    """
    # We start with the smallest stride, jumping
    # from rx channel to the other
    st_chan = itemsize
    # The second smallest jumps from one range sample to the next
    # so we have to jump to every nchan-th sample
    st_rg = st_chan * nchan
    # To move from one pattern to the next, we move to the next impulse
    st_pat = (nsamp + 1) * st_rg
    # To move in azimuth to the subsequent record with the same
    # pattern , we have to jump npat times a range record
    st_az = st_pat * npat
    return st_rg, st_az, st_chan, st_pat


def count_pat(TX_RX_SEQ):
    return len(TX_RX_SEQ.split('-'))


def load_raw(par_path, path, nchan=2):
    """
    This function loads a gamma raw dataset
    :param the path to the raw_par file:
    :param the path to the raw file:
    :return:
    """
    par = par_to_dict(par_path)
    nsamp = par['CHP_num_samp']
    npat = count_pat(par['TX_RX_SEQ'])
    # If multiplexed data, we have two channels
    if npat > 1:
        nchan = 2
    else:
        nchan = 1
    itemsize = _np.int16(1).itemsize
    bytes_per_record = (nsamp + 1) * nchan * itemsize
    filesize = _osp.getsize(path)
    raw = _np.memmap(path, dtype=type_mapping['SHORT INTEGER'], mode='r')
    nl_tot = int(filesize / bytes_per_record)
    sh = (nsamp + 1, nl_tot / npat,
          nchan, npat)
    stride = gpri_raw_strides(nsamp, nchan, npat, itemsize)
    raw_shp = _ast(raw, shape=sh, strides=stride).squeeze()
    return raw_shp, par


def default_slc_dict():
    """
    This function creates a default dict for the slc parameters of the
    gpri, that the user can then fill according to needs
    :return:
    """
    par = _od()
    par['title'] = ''
    par['sensor'] = 'GPRI 2'
    par['date'] = [0, 0, 0]
    par['start_time'] = 0
    par['center_time'] = 0
    par['end_time'] = 0
    par['azimuth_line_time'] = 0
    par['line_header_size'] = 0
    par['range_samples'] = 0
    par['azimuth_lines'] = 0
    par['range_looks'] = 1
    par['azimuth_looks'] = 1
    par['image_format'] = 'FCOMPLEX'
    par['image_geometry'] = 'SLANT_RANGE'
    par['range_scale_factor'] = 1
    par['azimuth_scale_factor'] = 1
    par['center_latitude'] = [0, 'degrees']
    par['center_longitude'] = [0, 'degrees']
    par['heading'] = [0, 'degrees']
    par['range_pixel_spacing'] = [0, 'm']
    par['azimuth_pixel_spacing'] = [0, 'm']
    par['near_range_slc'] = [0, 'm']
    par['center_range_slc'] = [0, 'm']
    par['far_range_slc'] = [0, 'm']
    par['first_slant_range_polynomial'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    par['center_slant_range_polynomial'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    par['last_slant_range_polynomial'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    par['incidence_angle'] = [0.0, 'degrees']
    par['azimuth_deskew'] = 'OFF'
    par['azimuth_angle'] = [0.0, 'degrees']
    par['radar_frequency'] = [0.0, 'Hz']
    par['adc_sampling_rate'] = [0.0, 'Hz']
    par['chirp_bandwidth'] = [0.0, 'Hz']
    par['prf'] = [0.0, 'Hz']
    par['azimuth_proc_bandwidth'] = [0.0, 'Hz']
    par['doppler_polynomial'] = [0.0, 0.0, 0.0, 0.0]
    par['doppler_poly_dot'] = [0.0, 0.0, 0.0, 0.0]
    par['doppler_poly_ddot'] = [0.0, 0.0, 0.0, 0.0]
    par['receiver_gain'] = [0.0, 'dB']
    par['calibration_gain'] = [0.0, 'dB']
    par['sar_to_earth_center'] = [0.0, 'm']
    par['earth_radius_below_sensor'] = [0.0, 'm']
    par['earth_semi_major_axis'] = [ra, 'm']
    par['earth_semi_minor_axis'] = [rb, 'm']
    par['number_of_state_vectors'] = 0
    par['GPRI_TX_mode'] = ''
    par['GPRI_TX_antenna'] = ''
    par['GPRI_RX_antenna'] = ''
    par['GPRI_az_start_angle'] = [0, 'degrees']
    par['GPRI_az_angle_step'] = [0, 'degrees']
    par['GPRI_ant_elev_angle'] = [0, 'degrees']
    par['GPRI_ref_north'] = 0
    par['GPRI_ref_east'] = 0
    par['GPRI_ref_alt'] = [0, 'm']
    par['GPRI_geoid'] = [0, 'm']
    par['GPRI_scan_heading'] = [0, 'degrees']
    return par


class rawParameters:
    """
    This class computes several raw parameters from
    a raw_par file
    """

    def __init__(self, raw_dict, raw):
        if hasattr(raw_dict, 'iteritems'):  # if the user passes a string, first load the corresponding dict
            pass
        else:
            raw_dict = par_to_dict(raw_dict)
        self.grp = _nt('GenericDict', raw_dict.keys())(**raw_dict)
        self.nsamp = self.grp.CHP_num_samp
        self.block_length = self.nsamp + 1
        self.chirp_duration = self.block_length / self.grp.ADC_sample_rate
        self.pn1 = _np.arange(self.nsamp / 2 + 1)  # list of slant range pixel numbers
        self.rps = (self.grp.ADC_sample_rate / self.nsamp * C / 2.) / self.grp.RF_chirp_rate  # range pixel spacing
        self.slr = (
                       self.pn1 * self.grp.ADC_sample_rate / self.nsamp * C / 2.) / self.grp.RF_chirp_rate + RANGE_OFFSET  # slant range for each sample
        self.scale = (abs(self.slr) / self.slr[self.nsamp / 8]) ** 1.5  # cubic range weighting in power
        self.ns_max = int(round(0.90 * self.nsamp / 2))  # default maximum number of range samples for this chirp
        self.tcycle = self.block_length / self.grp.ADC_sample_rate  # time/cycle
        self.dt = type_mapping['SHORT INTEGER']
        self.sizeof_data = _np.dtype(_np.int16).itemsize
        self.bytes_per_record = self.sizeof_data * self.block_length  # number of bytes per echo
        # Get file size
        try:
            self.filesize = _os.path.getsize(raw)
            self.nl_tot = int(self.filesize / (self.sizeof_data * self.block_length))
        except:
            try:
                self.nl_tot = raw.shape[1]
            except:
                pass
        # Number of lines
        # Stuff for angle
        if self.grp.STP_antenna_end != self.grp.STP_antenna_start:
            self.ang_acc = self.grp.TSC_acc_ramp_angle
            rate_max = self.grp.TSC_rotation_speed
            self.t_acc = self.grp.TSC_acc_ramp_time
            self.ang_per_tcycle = self.tcycle * self.grp.TSC_rotation_speed  # angular sweep/transmit cycle
        else:
            self.t_acc = 0.0
            self.ang_acc = 0.0
            rate_max = 0.0
            self.ang_per_tcycle = 0.0
        if self.grp.ADC_capture_time == 0.0:
            angc = abs(
                self.grp.antenna_end - self.grp.antenna_start) - 2 * self.ang_acc  # angular sweep of constant velocity
            tc = abs(angc / rate_max)  # duration of constant motion
            self.grp.capture_time = 2 * self.t_acc + tc  # total time for scanner to complete scan
            # Frequenct vector
        self.freq_vec = self.grp.RF_freq_min + _np.arange(self.grp.CHP_num_samp,
                                                          dtype=float) * self.grp.RF_chirp_rate / self.grp.ADC_sample_rate

    def compute_slc_parameters(self, kbeta=3.0, rmin=50, rmax=1000, dec=5,
                               zero=300):  # compute the slc parameters for a given set of input parameters
        self.rmax = self.ns_max * self.rps  # default maximum slant range
        self.win = _sig.kaiser(self.nsamp, kbeta)
        self.zero = zero
        self.win2 = _sig.hanning(
            2 * self.zero)  # window to remove transient at start of echo due to sawtooth frequency sweep
        self.ns_min = int(round(rmin / self.rps))  # round to the nearest range sample
        self.ns_out = (self.ns_max - self.ns_min) + 1
        self.rmin = self.ns_min * self.rps
        self.dec = dec
        self.nl_acc = int(self.t_acc / (self.tcycle * self.dec))
        # self.nl_tot = int(self.grp.ADC_capture_time/(self.tcycle))
        self.nl_tot_dec = self.nl_tot / self.dec
        self.nl_image = self.nl_tot_dec - 2 * self.nl_acc
        self.image_time = (self.nl_image - 1) * (self.tcycle * self.dec)
        if rmax != 0.0:  # check if greater than maximum value for the selected chirp
            if int(round(rmax / self.rps)) <= self.ns_max:
                self.ns_max = int(round(rmax / self.rps))
                self.rmax = self.ns_max * self.rps
            else:
                print(
                    "ERROR: requested maximum slant range exceeds maximum possible value with this chirp: {value:f<30}'".format(
                        self.rmax, ))

        self.ns_out = (self.ns_max - self.ns_min) + 1  # number of output samples
        # Compute antenna positions
        if self.grp.STP_antenna_end > self.grp.STP_antenna_start:  # clockwise rotation looking down the rotation axis
            self.az_start = self.grp.STP_antenna_start + self.ang_acc  # offset to start of constant motion + center-freq squint
        else:  # counter-clockwise
            self.az_start = self.grp.STP_antenna_start - self.ang_acc

    def fill_dict(self):
        """
        This function fills the slc dict with the correct image
        parameters
        :return:
        """
        image_time = (self.nl_image - 1) * (self.tcycle * self.dec)
        slc_dict = default_slc_dict()
        ts = self.grp.time_start
        ymd = (ts.split()[0]).split('-')
        hms_tz = (ts.split()[1]).split('+')  # split into HMS and time zone information
        hms = (hms_tz[0]).split(':')  # split HMS string using :
        sod = int(hms[0]) * 3600 + int(hms[1]) * 60 + float(hms[2])  # raw data starting time, seconds of day
        st0 = sod + self.nl_acc * self.tcycle * self.dec + \
              (self.dec / 2.0) * self.tcycle  # include time to center of decimation window
        az_step = self.ang_per_tcycle * self.dec
        prf = abs(1.0 / (self.tcycle * self.dec))
        seq = self.grp.TX_RX_SEQ
        fadc = C / (2. * self.rps)
        # Antenna elevation angle
        ant_elev = _np.deg2rad(self.grp.antenna_elevation)
        # Compute antenna position
        rx1_coord = [0., 0., 0.]
        rx2_coord = [0., 0., 0.]
        tx_coord = [0., 0., 0.]
        #
        # Topsome receiver
        rx1_coord[0] = xoff + ant_radius * _np.cos(
            ant_elev)  # local coordinates of the tower: x,y,z, boresight is along +X axis, +Z is up
        rx1_coord[1] = 0.0  # +Y is to the right when looking in the direction of +X
        rx1_coord[2] = rx1_dz[seq[1]] + ant_radius * _np.sin(
            ant_elev)  # up is Z, all antennas have the same elevation angle!
        # Bottomsome receiver
        rx2_coord[0] = xoff + ant_radius * _np.cos(ant_elev)
        rx2_coord[1] = 0.0
        rx2_coord[2] = rx2_dz[seq[2]] + ant_radius * _np.sin(ant_elev)
        tx_coord[0] = xoff + ant_radius * _np.cos(ant_elev)
        tx_coord[1] = 0.0
        tx_coord[2] = tx_dz[seq[0]] + ant_radius * _np.sin(ant_elev)
        chan_name = 'CH1 lower' if seq[3] == 'l' else 'CH2 upper'
        slc_dict['title'] = ts + ' ' + chan_name
        slc_dict['date'] = [ymd[0], ymd[1], ymd[2]]
        slc_dict['start_time'] = [st0, 's']
        slc_dict['center_time'] = [st0 + image_time / 2, 's']
        slc_dict['end_time'] = [st0 + image_time, 's']
        slc_dict['range_samples'] = self.ns_out
        slc_dict['azimuth_lines'] = self.nl_tot_dec - 2 * self.nl_acc
        slc_dict['range_pixel_spacing'] = [self.rps, 'm']
        slc_dict['azimuth_line_time'] = [self.tcycle * self.dec, 's']
        slc_dict['near_range_slc'] = [self.rmin, 'm']
        slc_dict['center_range_slc'] = [(self.rmin + self.rmax) / 2, 'm']
        slc_dict['far_range_slc'] = [self.rmax, 'm']
        slc_dict['radar_frequency'] = [self.grp.RF_center_freq, 'Hz']
        slc_dict['adc_sampling_rate'] = [fadc, 'Hz']
        slc_dict['prf'] = [prf, 'Hz']
        slc_dict['chirp_bandwidth'] = self.grp.RF_freq_max - self.grp.RF_freq_min
        slc_dict['receiver_gain'] = [60 - self.grp.IMA_atten_dB, 'dB']
        slc_dict['GPRI_TX_mode'] = self.grp.TX_mode
        slc_dict['GPRI_TX_antenna'] = seq[0]
        slc_dict['GPRI_RX_antenna'] = seq[1] + seq[3]
        slc_dict['GPRI_tx_coord'] = [tx_coord[0], tx_coord[1], tx_coord[2], 'm', 'm', 'm']
        slc_dict['GPRI_rx1_coord'] = [rx1_coord[0], rx1_coord[1], rx1_coord[2], 'm', 'm', 'm']
        slc_dict['GPRI_rx2_coord'] = [rx2_coord[0], rx2_coord[1], rx2_coord[2], 'm', 'm', 'm']
        slc_dict['GPRI_az_start_angle'] = [self.az_start, 'degrees']
        slc_dict['GPRI_az_angle_step'] = [az_step, 'degrees']
        slc_dict['GPRI_ant_elev_angle'] = [self.grp.antenna_elevation, 'degrees']
        slc_dict['GPRI_ref_north'] = [self.grp.geographic_coordinates[0], 'degrees']
        slc_dict['GPRI_ref_east'] = [self.grp.geographic_coordinates[1], 'degrees']
        slc_dict['GPRI_ref_alt'] = [self.grp.geographic_coordinates[2], 'm']
        slc_dict['GPRI_geoid'] = [self.grp.geographic_coordinates[3], 'm']
        return slc_dict


class rawData(_np.ndarray):
    def __array_wrap__(self, out_arr):
        # This is just a lazy
        # _array_wrap_ that
        # copies properties from
        # the input object
        out_arr = out_arr.view(type(self))
        try:
            out_arr.__dict__ = self.__dict__.copy()
        except:
            pass
        return out_arr

    def __array_finalize__(self, obj):
        if obj is None: return
        if hasattr(obj, '__dict__'):
            self.__dict__ = _cp.deepcopy(obj.__dict__)

    def __new__(cls, *args, **kwargs):
        if 'channel_mapping' not in kwargs:
            channel_mapping = {'TX_A_position': 0, 'TX_B_position': 0.125,
                               'RX_Au_position': 0.475, 'RX_Al_position': 0.725,
                               'RX_Bu_position': 0.6, 'RX_Bl_position': 0.85}
        else:
            channel_mapping = kwargs['channel_mapping']
        data, par_dict = load_raw(args[0], args[1])
        obj = data.view(cls)
        obj.__dict__ = _cp.deepcopy(par_dict)
        obj.nsamp = obj.CHP_num_samp
        obj.block_length = obj.CHP_num_samp + 1
        obj.chirp_duration = obj.block_length / obj.ADC_sample_rate
        obj.pn1 = _np.arange(obj.nsamp / 2 + 1)  # list of slant range pixel numbers
        obj.rps = (obj.ADC_sample_rate / obj.nsamp * C / 2.) / obj.RF_chirp_rate  # range pixel spacing
        obj.slr = (
                      obj.pn1 * obj.ADC_sample_rate / obj.nsamp * C / 2.) / obj.RF_chirp_rate + RANGE_OFFSET  # slant range for each sample
        obj.scale = (abs(obj.slr) / obj.slr[obj.nsamp / 8]) ** 1.5  # cubic range weighting in power
        obj.ns_max = int(round(0.90 * obj.nsamp / 2))  # default maximum number of range samples for this chirp
        # obj.tcycle = (obj.block_length) / obj.ADC_sample_rate  # time/cycle
        obj.dt = type_mapping['SHORT INTEGER']
        obj.sizeof_data = _np.dtype(_np.int16).itemsize
        obj.bytes_per_record = obj.sizeof_data * obj.block_length  # number of bytes per echo
        obj.mapping_dict = channel_mapping
        return obj

    def tofile(*args):
        self = args[0]
        # In this case, we want to write both parameters and binary file
        if len(args) is 3:
            write_dataset(self, self.__dict__, args[1], args[2])

    def extract_channel(self, pat, ant):
        if self.npats > 1:
            chan_idx = self.channel_index(pat, ant)
            chan = self[:, :, chan_idx[0], chan_idx[1]]
            chan = self.__array_wrap__(chan)
            # chan.tcycle = chan.tcycle / self.npats
            chan.GPRI_TX_antenna_position = self.mapping_dict['TX_' + pat[0] + "_position"]
            chan.GPRI_RX_antenna_position = self.mapping_dict['RX_' + pat[1] + ant + "_position"]
            chan.ADC_capture_time = self.ADC_capture_time / self.npats
            chan.TSC_rotation_speed = self.TSC_rotation_speed * self.npats
            chan.STP_rotation_speed = self.STP_rotation_speed * self.npats
            chan.TSC_acc_ramp_time = self.TSC_acc_ramp_time / self.npats
            chan.TX_mode = None
            chan.TX_RX_SEQ = pat + ant
            print(chan.shape)
            return chan
        else:
            raise IndexError('This dataset only has one channel')

    def channel_index(self, pat, ant):
        chan_list = self.TX_RX_SEQ.split('-')
        chan_idx = chan_list.index(pat)
        print(ant)
        return [ant_map[ant], chan_idx]

    # All proprerties that depend directly or indirectly on the number of patterns
    # are computed on the fly using property decorators

    @property
    def npats(self):
        return len(self.TX_RX_SEQ.split('-'))

    @property
    def azspacing(self):
        return self.tcycle * self.STP_rotation_speed * self.npats

    @property
    def freqvec(self):
        return self.RF_freq_min + _np.arange(self.CHP_num_samp,
                                             dtype=_np.double) * self.RF_chirp_rate / self.ADC_sample_rate

    @property
    def nl_tot(self):
        return self.shape[1]

    @property
    def tcycle(self):
        return self.shape[0] * 1 / self.ADC_sample_rate

    @property
    def ang_per_tcycle(self):
        return self.tcycle * self.TSC_rotation_speed

    @property
    def capture_time(self):
        if self.ADC_capture_time == 0:
            angc = abs(
                self.STP_antenna_end - self.STP_antenna_start) - 2 * self.TSC_acc_ramp_time  # angular sweep of constant velocity
            tc = abs(angc / self.TSC_rotation_speed)  # duration of constant motion
            capture_time = 2 * self.self.grp.TSC_acc_ramp_time + tc  # total time for scanner to complete scan
        else:
            capture_time = self.ADC_capture_time
        return capture_time

    @property
    def start_time(self):
        return str(self.time_start)

    def compute_slc_parameters(self, kbeta=3.0, rmin=50, dec=5, zero=300,
                               **kwargs):  # compute the slc parameters for a given set of input parameters
        if 'rmax' not in kwargs:
            rmax = self.ns_max * self.rps  # default maximum slant range
        else:
            rmax = kwargs.get('rmax')
        self.win = _sig.kaiser(self.nsamp, kbeta)
        self.zero = zero
        self.win2 = _sig.hanning(
            2 * self.zero)  # window to remove transient at start of echo due to sawtooth frequency sweep
        self.ns_min = int(round(rmin / self.rps))  # round to the nearest range sample
        self.ns_max = int(round(0.90 * self.nsamp / 2))  # default maximum number of range samples for this chirp
        self.ns_out = (self.ns_max - self.ns_min) + 1
        self.rmin = self.ns_min * self.rps
        self.dec = dec
        self.nl_acc = int(self.TSC_acc_ramp_time / (self.tcycle * self.dec))
        # self.nl_tot = int(self.grp.ADC_capture_time/(self.tcycle))
        self.nl_tot_dec = int(self.nl_tot / self.dec)
        self.nl_image = self.nl_tot_dec - 2 * self.nl_acc
        self.image_time = (self.nl_image - 1) * (self.tcycle * self.dec)
        if rmax != 0.0:  # check if greater than maximum value for the selected chirp
            if int(round(rmax / self.rps)) <= self.ns_max:
                self.ns_max = int(round(rmax / self.rps))
                self.rmax = self.ns_max * self.rps
            else:
                print(
                    "ERROR: requested maximum slant range exceeds maximum possible value with this chirp: {value:f<30}'".format(
                        self.rmax_chirp, ))

        self.ns_out = (self.ns_max - self.ns_min) + 1  # number of output samples
        # Compute antenna positions
        if self.STP_antenna_end > self.STP_antenna_start:  # clockwise rotation looking down the rotation axis
            self.az_start = self.STP_antenna_start + self.TSC_acc_ramp_angle  # offset to start of constant motion + center-freq squint
        else:  # counter-clockwise
            self.az_start = self.STP_antenna_start - self.TSC_acc_ramp_angle

    def fill_dict(self):
        """
        This function fills the slc dict with the correct image
        parameters
        :return:
        """
        image_time = (self.nl_image - 1) * (self.tcycle * self.dec)
        slc_dict = default_slc_dict()
        ts = self.time_start
        ymd = (ts.split()[0]).split('-')
        hms_tz = (ts.split()[1]).split('+')  # split into HMS and time zone information
        hms = (hms_tz[0]).split(':')  # split HMS string using :
        sod = int(hms[0]) * 3600 + int(hms[1]) * 60 + float(hms[2])  # raw data starting time, seconds of day
        st0 = sod + self.nl_acc * self.tcycle * self.dec + \
              (self.dec / 2.0) * self.tcycle  # include time to center of decimation window
        az_step = self.ang_per_tcycle * self.dec
        prf = abs(1.0 / (self.tcycle * self.dec))
        seq = self.TX_RX_SEQ
        fadc = C / (2. * self.rps)
        # Antenna elevation angle
        ant_elev = _np.deg2rad(self.antenna_elevation)
        # Compute antenna position
        rx1_coord = [0., 0., 0.]
        rx2_coord = [0., 0., 0.]
        tx_coord = [0., 0., 0.]
        #
        # Topsome receiver
        rx1_coord[0] = xoff + ant_radius * _np.cos(
            ant_elev)  # local coordinates of the tower: x,y,z, boresight is along +X axis, +Z is up
        rx1_coord[1] = 0.0  # +Y is to the right when looking in the direction of +X
        rx1_coord[2] = rx1_dz[seq[1]] + ant_radius * _np.sin(
            ant_elev)  # up is Z, all antennas have the same elevation angle!
        # Bottomsome receiver
        rx2_coord[0] = xoff + ant_radius * _np.cos(ant_elev)
        rx2_coord[1] = 0.0
        rx2_coord[2] = rx2_dz[seq[2]] + ant_radius * _np.sin(ant_elev)
        tx_coord[0] = xoff + ant_radius * _np.cos(ant_elev)
        tx_coord[1] = 0.0
        tx_coord[2] = tx_dz[seq[0]] + ant_radius * _np.sin(ant_elev)
        chan_name = 'CH1 lower' if seq[3] == 'l' else 'CH2 upper'
        slc_dict['title'] = ts + ' ' + chan_name
        slc_dict['date'] = [ymd[0], ymd[1], ymd[2]]
        slc_dict['start_time'] = [st0, 's']
        slc_dict['center_time'] = [st0 + image_time / 2, 's']
        slc_dict['end_time'] = [st0 + image_time, 's']
        slc_dict['range_samples'] = self.ns_out
        slc_dict['azimuth_lines'] = self.nl_tot_dec - 2 * self.nl_acc
        slc_dict['range_pixel_spacing'] = [self.rps, 'm']
        slc_dict['azimuth_line_time'] = [self.tcycle * self.dec, 's']
        slc_dict['near_range_slc'] = [self.rmin, 'm']
        slc_dict['center_range_slc'] = [(self.rmin + self.rmax) / 2, 'm']
        slc_dict['far_range_slc'] = [self.rmax, 'm']
        slc_dict['radar_frequency'] = [self.RF_center_freq, 'Hz']
        slc_dict['adc_sampling_rate'] = [fadc, 'Hz']
        slc_dict['prf'] = [prf, 'Hz']
        slc_dict['chirp_bandwidth'] = self.RF_freq_max - self.RF_freq_min
        slc_dict['receiver_gain'] = [60 - self.IMA_atten_dB, 'dB']
        slc_dict['GPRI_TX_mode'] = self.TX_mode
        slc_dict['GPRI_TX_antenna'] = seq[0]
        slc_dict['GPRI_RX_antenna'] = seq[1] + seq[3]
        slc_dict['GPRI_tx_coord'] = [tx_coord[0], tx_coord[1], tx_coord[2], 'm', 'm', 'm']
        slc_dict['GPRI_rx1_coord'] = [rx1_coord[0], rx1_coord[1], rx1_coord[2], 'm', 'm', 'm']
        slc_dict['GPRI_rx2_coord'] = [rx2_coord[0], rx2_coord[1], rx2_coord[2], 'm', 'm', 'm']
        slc_dict['GPRI_az_start_angle'] = [self.az_start, 'degrees']
        slc_dict['GPRI_az_angle_step'] = [az_step, 'degrees']
        slc_dict['GPRI_ant_elev_angle'] = [self.antenna_elevation, 'degrees']
        slc_dict['GPRI_ref_north'] = [self.geographic_coordinates[0], 'degrees']
        slc_dict['GPRI_ref_east'] = [self.geographic_coordinates[1], 'degrees']
        slc_dict['GPRI_ref_alt'] = [self.geographic_coordinates[2], 'm']
        slc_dict['GPRI_geoid'] = [self.geographic_coordinates[3], 'm']
        return slc_dict


def model_squint(freq_vec):
    return squint_angle(freq_vec, KU_WIDTH, KU_DZ, k=1.0 / 2.0)


def linear_squint(freq_vec, sq_parameters):
    return _np.polynomial.polynomial.polyval(freq_vec, [0, sq_parameters])


def correct_squint(raw_channel, squint_function=linear_squint, squint_rate=4.2e-9):
    # We require a function to compute the squint angle
    squint_vec = squint_function(raw_channel.freqvec, squint_rate)
    squint_vec = squint_vec / raw_channel.ang_per_tcycle
    squint_vec = squint_vec - squint_vec[
        raw_channel.freqvec.shape[0] / 2]  # In addition, we correct for the beam motion during the chirp
    rotation_squint = _np.linspace(0, raw_channel.tcycle,
                                   raw_channel.nsamp) * raw_channel.TSC_rotation_speed / raw_channel.ang_per_tcycle
    # Normal angle vector
    angle_vec = _np.arange(raw_channel.shape[1])
    # We do not correct the squint of the first sample
    squint_vec = _np.insert(squint_vec, 0, 0)
    rotation_squint = _np.insert(rotation_squint, 0, 0)
    # Interpolated raw channel
    raw_channel_interp = _np.zeros_like(raw_channel)
    for idx in range(0, raw_channel.shape[0]):
        az_new = angle_vec + squint_vec[idx] - rotation_squint[idx]
        if idx % 500 == 0:
            print_str = "interp sample: {idx}, ,shift: {sh} samples".format(idx=idx, sh=az_new[0] - angle_vec[0])
            print(print_str)
        raw_channel_interp[idx, :] = _np.interp(az_new, angle_vec, raw_channel[idx, :], left=0.0, right=0.0)
    raw_channel_interp.__array_wrap__(raw_channel)
    return raw_channel_interp


def correct_squint_in_SLC(SLC, squint_function=linear_squint, squint_rate=4.2e-9):
    SLC_corr = SLC * 1
    rawdata = _np.zeros((SLC.shape[0] / 2 - 1, SLC.shape[1]))
    rawdata_corr = rawdata * 1
    # Convert the data into raw samples
    for idx_line in range(SLC.shape[1]):
        rawdata[:, idx_line] = _np.fft.irfft(SLC[:, idx_line])
    # Now correct the squint
    freqvec = SLC.radar_frequency[0] + _np.linspace(-SLC.chirp_bandwidth[0] / 2, -SLC.chirp_bandwidth[0] / 2,
                                                    rawdata.shape[0])
    squint_vec = squint_function(freqvec, squint_rate)
    squint_vec = squint_vec / SLC.GPRI_az_angle_step[0]
    squint_vec = squint_vec - squint_vec[
        freqvec.shape[0] / 2]  # In addition, we correct for the beam motion during the chirp
    # Normal angle vector
    angle_vec = _np.arange(SLC.shape[1])
    # Correct by interpolation
    for idx_freq in range(rawdata.shape[0]):
        az_new = angle_vec + squint_vec[idx_freq]
        rawdata_corr[idx_freq, :] = _np.interp(az_new, angle_vec, SLC[idx_freq, :], left=0.0, right=0.0)
    # Now range compress again (this function is really boring
    for idx_line in range(SLC.shape[1]):
        SLC_corr[:, idx_line] = _np.fft.rfft(rawdata_corr)
    return SLC_corr


def range_compression(rawdata, rmin=50, rmax=None, kbeta=3.0, dec=1, zero=300, f_c=None, bw=66e6, rvp_corr=False,
                      scale=True):
    rvp = _np.exp(1j * 4. * _np.pi * rawdata.RF_chirp_rate * (rawdata.slr / C) ** 2)
    rawdata.compute_slc_parameters(kbeta=kbeta, rmin=rmin, rmax=rmax, zero=zero, dec=dec)
    # Construct a filter
    if f_c is None:
        filt = _np.ones(rawdata.nsamp)
    # else:
    #     filt = self.subband_filter(f_c, bw=bw)
    arr_compr = _np.zeros((rawdata.ns_max - rawdata.ns_min + 1, rawdata.nl_tot_dec), dtype=_np.complex64)
    fshift = _np.ones(rawdata.nsamp / 2 + 1)
    fshift[1::2] = -1
    # For each azimuth
    for idx_az in range(0, rawdata.nl_tot_dec):
        # Decimated pulse
        dec_pulse = _np.zeros(rawdata.block_length, dtype=_np.float32)
        for idx_dec in range(rawdata.dec):
            current_idx = idx_az * rawdata.dec + idx_dec
            current_idx_1 = idx_az + idx_dec * rawdata.dec
            if rawdata.dt == _np.dtype(_np.int16) or rawdata.dt == type_mapping['SHORT INTEGER']:
                current_data = rawdata[:, current_idx].astype(_np.float32) / TSF
            else:
                current_data = rawdata[:, current_idx].astype(_np.float32)
            if current_idx % 1000 == 0:
                print('Accessing azimuth index: {} '.format(current_idx))
            try:
                dec_pulse = dec_pulse + current_data
            except IndexError as err:
                print(err.message)
                break
        if rawdata.zero > 0:
            dec_pulse[0:rawdata.zero] = dec_pulse[0:rawdata.zero] * rawdata.win2[0:rawdata.zero]
            dec_pulse[-rawdata.zero:] = dec_pulse[-rawdata.zero:] * rawdata.win2[-rawdata.zero:]
        # Emilinate the first sample, as it is used to jump back to the start freq
        line_comp = _np.fft.rfft(dec_pulse[1::] * filt / rawdata.dec * rawdata.win) * fshift
        if rvp_corr:
            line_comp = line_comp * rvp
        # Decide if applying the range scale factor or not
        if scale:
            scale_factor = rawdata.scale[rawdata.ns_min:rawdata.ns_max + 1]
        else:
            scale_factor = 1
        arr_compr[:, idx_az] = (line_comp[rawdata.ns_min:rawdata.ns_max + 1].conj() * scale_factor).astype('complex64')
    # Remove lines used for rotational acceleration
    arr_compr = arr_compr[:, rawdata.nl_acc:rawdata.nl_image + rawdata.nl_acc:]
    slc_dict = rawdata.fill_dict()
    slc_compr = gammaDataset(slc_dict, arr_compr)
    return slc_compr


def lamg(freq, w):
    """
    This function computes the wavelength in waveguide for the TE10 mode
    """
    la = lam(freq)
    return la / _np.sqrt(1.0 - (la / (2 * w)) ** 2)  # wavelength in WG-62 waveguide


# lambda in freespace
def lam(freq):
    """
    This function computes the wavelength in freespace
    """
    return C / freq


def squint_angle(freq, w, s, k=0):
    """
    This function computes the direction of the main lobe of a slotted
    waveguide antenna as a function of the frequency, the size and the slot spacing.
    It supposes a waveguide for the TE10 mode
    """
    sq_ang = _np.pi / 2.0 - _np.arccos(lam(freq) / lamg(freq, w) - k * lam(freq) / s)
    # sq_ang = 90 - _np.rad2deg(_np.arccos(lam(freq) * k / s + 2 * _np.pi * lam(freq)/lamg(freq, w)))
    # dphi = _np.pi * (2. * s / lamg(freq, w) - 1.0)  # antenna phase taper to generate squint
    # sq_ang_1 = _np.rad2deg(_np.arcsin(lam(freq) * dphi / (2. * _np.pi * s)))  # azimuth beam squint angle
    return _np.rad2deg(sq_ang)


def load_segment(file, shape, xmin, xmax, ymin, ymax, dtype=type_mapping['FCOMPLEX']):
    """
    This function load a segment from a file that represents
    a 2D array (image)
    Parameters
    ----------
    file : string, file
        The file to load from
    shape : iterable
        The shape of the file
    xmin : int
        The starting x location of the window to extract
    xmax : int
        The ending x location of the window to extract (-1 for until end)
    ymin : int
        The starting y location of the window to extract
    ymax : int
        The ending y location of the window to extract (-1 for unntil end)
    dtype : dtype
        The type of the data which is to be loaded

    Returns
    -------
    A 2d numpy array of values with the type `dtype`

    """

    # x corresponds to range, y to azimuth
    # Seek before starting memmap (seek as many records as ymin)
    inital_seek = int(ymin * shape[0] * dtype.itemsize)
    # How many records to load
    nrecords = int(shape[0] * (ymax - ymin))
    with open(file, 'rb') as input_file:
        input_file.seek(inital_seek)
        # Load the desired records (this automagically provides y slicing
        selected_records = _np.fromfile(input_file, dtype=dtype, count=nrecords)
        # Reshape with the record length
        output_shape = ((ymax - ymin), shape[0])
        selected_records = selected_records.reshape(output_shape).T
        output_slice = selected_records[xmin:xmax, :]
        return output_slice


def ant_pos(t0, angle_start, angle_end, gear_ratio, max_speed):
    # return the current rotation speed and angle given a time since the start of rotation
    # v1.0 clw 22-Oct-2010
    # v1.1 clw 7-Nov-2011 correct rotation rates
    #
    #  t0  		current time after rotation start
    #  angle_start  starting antenna rotation angle
    #  angle_end    ending antenna rotation angle
    #  gear_ratio   gear ratio (72:1 or 80:1) specify 72 or 80 as the argument
    #  max_speed	maximum speed in steps of 0.5 from 0.5 to 10.0 degrees/sec sent to TSCC
    # 		The exact speed is determined from a lookup table
    #
    #  returns the current exact rotation speed and angle,
    #  and times for start and end of constant velocity motion

    if gear_ratio == 72:  # 72:1 gear ratio
        rate = (
            0.50080, 1.00160, 1.50240, 2.00321, 2.50401, 3.00481, 3.48772, 3.98597, 4.48994, 5.00801, 5.50176, 6.00962,
            6.51042, 7.00673, 7.51202, 8.01282, 8.49185, 8.97989, 9.52744, 10.01603)
        #   rate = (0.50080, 1.00677, 1.51406, 2.01354, 2.52017, 3.02811, 3.51916, 4.02708, 4.54217, 5.07307, 5.58038, 6.10354, 6.53766, 7.03830, 7.54832, 8.05413, 8.53826, 9.03180, 9.58590, 10.08066)
        ramp = (
            0.00000, 0.01250, 0.03750, 0.07500, 0.12500, 0.18750, 0.26250, 0.35000, 0.45000, 0.56250, 0.68750, 0.82500,
            0.97500, 1.13750, 1.31250, 1.50000, 1.70000, 1.91250, 2.13750, 2.37500)
        tstep = 0.025  # time step for velocities
    else:
        if gear_ratio == 80:  # 80:1 gear ratio
            rate = (
                0.49938, 0.99876, 1.50240, 1.99751, 2.49335, 3.00481, 3.51563, 3.99503, 4.50721, 5.02232, 5.49316,
                5.95869,
                6.51042, 6.99627, 7.48005, 7.99006, 8.52273, 9.01442, 9.50169, 9.97340)
            #     rate = (0.50224, 1.00447, 1.51537, 2.00894, 2.51117, 3.03072, 3.55115, 4.04096, 4.56577, 5.09513, 5.58038, 6.06145, 6.54070, 7.03126, 7.52006, 8.03572, 8.57470, 9.07259, 9.56634, 10.04465)
            ramp = (
                0.00000, 0.01125, 0.03375, 0.06750, 0.11250, 0.16875, 0.23625, 0.31500, 0.40500, 0.50625, 0.61875,
                0.74250,
                0.87750, 1.02375, 1.18125, 1.35000, 1.53000, 1.72125, 1.92375, 2.13750)
            tstep = 0.0225  # time step for velocities
        else:
            _sys.exit(-1)

    if max_speed > 10 or max_speed < 0:
        raise SystemExit

    ix = int(max_speed / 0.5 - 1)  # num steps of approx 0.5 deg/s to max speed, initial velocity is approx 0.5 deg/s
    if ix == -1:  # tower is stationary
        return 0., 0., 0., 0., 0.

    t_acc = ix * tstep  # time interval for acceleration to max_speed
    ang_acc = ramp[ix]  # angle at the end of the acceleration phase
    rate_max = rate[ix]  # actual velocity at the end of the acceleration phase
    sweep = angle_end - angle_start  # total angular sweep

    if abs(sweep) < (2 * ang_acc):  # check if the angle is less than 2 * the motion required for acceleration to max
        t_acc = 0.0
        ang_acc = 0.0
        rate_max = rate[0]

    angc = abs(sweep) - 2 * ang_acc  # angular sweep of constant velocity
    tc = angc / rate_max  # duration of constant motion
    t_dec = t_acc + tc  # time at the start of decceleration
    ang_dec = ang_acc + angc
    t_total = 2 * t_acc + tc  # total time
    #  print 't_acc: %.5f  ang_acc: %.5f  tc: %.5f  ang_c: %.5f  t_dec: %.5f  ang_dec: %.5f  t_total: %.5f'%(t_acc, ang_acc, tc, angc, t_dec, ang_dec, t_total)

    if t0 <= t_acc:  # acceleration phase
        itx = int(t0 / tstep)
        rate1 = rate[itx]
        ramp1 = ramp[itx]
        td = t0 - itx * tstep

        if angle_end > angle_start:
            ang = angle_start + ramp1 + td * rate1
        else:
            ang = angle_start - (ramp1 + td * rate1)
        return ang, rate1, t_acc, ang_acc, rate_max

    if t_acc < t0 <= t_dec:  # constant velocity phase
        if angle_end > angle_start:
            ang = angle_start + (ang_acc + (t0 - t_acc) * rate_max)
        else:
            ang = angle_start - (ang_acc + (t0 - t_acc) * rate_max)
        return ang, rate_max, t_acc, ang_acc, rate_max

    if t_dec < t0 <= t_total:  # decceleration phase
        td = (t0 - t_dec)  # time since start of deceleration
        itx = ix - (int(td / tstep) + 1)  # already deccelerating

        rate1 = rate[itx]
        ramp1 = ramp[ix] - ramp[itx + 1]

        if angle_end > angle_start:
            ang = angle_start + ang_acc + angc + ramp1 + (td - int(td / tstep) * tstep) * rate1
        else:
            ang = angle_start - (ang_acc + angc + ramp1 + (td - int(td / tstep) * tstep) * rate1)

        return ang, rate1, t_acc, ang_acc, rate_max

    if t0 >= t_total:
        return angle_end, 0.0, t_acc, ang_acc, rate_max

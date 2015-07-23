# -*- coding: utf-8 -*-
"""
Created on Thu May 15 13:51:30 2014

This module contains classes and function to deal with Gamma file formats


@author: baffelli
"""
import os.path as _osp
from osgeo import osr as _osr
import mmap as _mm
import copy as _cp
import numbers as _num

import tempfile as _tf

import numpy as _np
from numpy.lib.stride_tricks import as_strided as _ast
import scipy as _sp



# This dict defines the mapping
# between the gamma datasets and numpy
type_mapping = {
    'FCOMPLEX': _np.dtype('>c8'),
    'SCOMPLEX': _np.dtype('>c4'),
    'FLOAT': _np.dtype('>f4'),
    'SHORT INTEGER': _np.dtype('>i2'),
    'INTEGER*2': _np.dtype('<i2'),
    'REAL*4': _np.dtype('<f4')
}

# This dict defines the mapping
# between channels in the file name and the matrix
channel_dict = {
    'HH': (0, 0),
    'HV': (0, 1),
    'VH': (1, 0),
    'VV': (1, 1),
}


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


class gammaDataset(_np.ndarray):
    def __new__(*args, **kwargs):
        cls = args[0]
        if len(args) == 3:
            par_path = args[1]
            bin_path = args[2]
            memmap = kwargs.get('memmap', False)
            image, par_dict = load_dataset(par_path, bin_path, memmap=memmap)
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
        if type(item) is str:
            try:
                sl_mat = channel_dict[item]
                sl = (Ellipsis,) * (self.ndim - 2) + sl_mat
            except KeyError:
                raise IndexError('This channel does not exist')
        else:
            sl = item
        new_obj_1 = (super(gammaDataset, self).__getitem__(sl)).view(type(self))
        if hasattr(new_obj_1, 'near_range_slc'):
            # Construct temporary azimuth and  range vectors
            az_vec = self.az_vec * 1
            r_vec = self.r_vec * 1
            r_0 = self.az_vec[0]
            az_0 = self.r_vec[0]
            az_spac = self.GPRI_az_angle_step[0] * 1
            r_spac = self.range_pixel_spacing[0] * 1
            # Passing only number, slice along first dim only
            if isinstance(sl, _num.Number):
                az_0 = az_vec[sl]
                r_0 = self.near_range_slc[0] * 1
                az_spac = self.GPRI_az_angle_step[0] * 1
                r_spac = self.range_pixel_spacing[0] * 1
            # Tuple of slices (or integers)
            elif hasattr(sl, '__contains__'):
                # By taking the first element, we automatically have
                # the correct data
                az_vec_sl = az_vec[sl[1]]
                r_vec_sl = r_vec[sl[0]]
                # THe result of slicing
                # could be a number or an array
                if hasattr(az_vec_sl, '__contains__'):
                    if len(az_vec_sl) > 1:
                        az_spac = az_vec_sl[1] - az_vec_sl[0]
                    else:
                        az_spac = az_spac
                    az_0 = az_vec_sl[0]
                else:
                    az_0 = az_vec_sl
                    az_spac = self.GPRI_az_angle_step[0] * 1
                if hasattr(r_vec_sl, '__contains__'):
                    if len(r_vec_sl) > 1:
                        r_spac = r_vec_sl[1] - r_vec_sl[0]
                    else:
                        r_spac = r_spac
                    r_spac = r_vec_sl[1] - r_vec_sl[0]
                    r_0 = r_vec_sl[0]
                else:
                    r_spac = self.range_pixel_spacing[0] * 1
                    r_0 = r_vec_sl
            new_obj_1.GPRI_az_start_angle[0] = az_0
            new_obj_1.near_range_slc[0] = r_0
            new_obj_1.GPRI_az_angle_step[0] = az_spac
            new_obj_1.range_pixel_spacing[0] = r_spac
        return new_obj_1

    def tofile(*args):
        self = args[0]
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

    def rvec(obj):
        return obj.__dict__['near_range_slc'][0] + _np.arange(obj.__dict__['range_samples']) * \
                                                   obj.__dict__['range_pixel_spacing'][0]

    def azvec(obj):
        return obj.__dict__['GPRI_az_start_angle'][0] + _np.arange(obj.__dict__['azimuth_lines']) * \
                                                        obj.__dict__['GPRI_az_angle_step'][0]

    r_vec = property(rvec)
    az_vec = property(azvec)


def par_to_dict(par_file):
    """
    This function converts a gamma '.par' file into
    a dict of parameters
    :param par_file:
    A string containing the path to the parameter file
    :return:
    A dict of parameters
    """
    par_dict = {}
    with open(par_file, 'r') as fin:
        # Skip first line
        fin.readline()
        for line in fin:
            if line:
                split_array = line.split(':')
                if len(split_array) > 1:
                    key = split_array[0]
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
        for key, par in par_dict.iteritems():
            out = str(key) + ":" + '\t'
            if isinstance(par, basestring):
                out = out + par
            else:
                try:

                    for p in par:
                        out = out + '\t' + str(p)
                except TypeError:
                    out = out + str(par)
            fout.write(out + '\n')


def load_dataset(par_file, bin_file, memmap=True):
    par_dict = par_to_dict(par_file)
    # Map type to gamma
    try:
        dt = type_mapping[par_dict['image_format']]
    except:
        try:
            dt = type_mapping[par_dict['data_format']]
        except:
            raise KeyError("This file does not contain datatype specification in a known format")
    try:
        shape = (par_dict['range_samples'],
                 par_dict['azimuth_lines'])
    except KeyError:
        # We dont have a SAR image,
        # try as it were a DEM
        try:
            shape = (par_dict['nlines'],
                     par_dict['width'])
        except:
            # Last try
            # interferogram
            try:
                shape = (par_dict['interferogram_width'], par_dict['interferogram_azimuth_lines'])
            except:
                raise KeyError("This file does not contain data shape specification in a known format")
    shape = shape[::-1]
    if memmap:
        with open(bin_file, 'rb') as mmp:
            buffer = _mm.mmap(mmp.fileno(), 0, prot=_mm.PROT_READ)
            d_image = _np.ndarray(shape, dt, buffer).T
    else:
        d_image = _np.fromfile(bin_file, dtype=dt).reshape(shape).T
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


# TODO make compatible with other datums
def geotif_to_dem(gt, par_path, bin_path):
    """
    This function converts a gdal dataset
    DEM into a gamma format pair
    of binary DEM and parameter file
    """
    DEM = gt.ReadAsArray()
    GT = gt.GetGeoTransform()
    srs = _osr.SpatialReference()
    srs.ImportFromWkt(gt.GetProjection())
    d = {}
    # FOrmat information
    if _np.issubdtype(DEM.dtype, _np.int32):
        DEM = DEM.astype(_np.int16)
        d['data_format'] = 'INTEGER*2'
    elif _np.issubdtype(DEM.dtype, _np.int16):
        d['data_format'] = 'INTEGER*2'
    elif _np.issubdtype(DEM.dtype, _np.float32):
        d['data_format'] = 'REAL*4'
    # Geotransform information
    d['DEM_scale'] = 1.0
    d['DEM_projection'] = 'OMCH'
    d['projection_name'] = 'OM - Switzerland'
    d['corner_north'] = GT[3]
    d['corner_east'] = GT[0]
    d['width'] = gt.RasterXSize
    d['nlines'] = gt.RasterYSize
    d['post_north'] = GT[5]
    d['post_east'] = GT[1]
    # Ellipsoid information
    d['ellipsoid_name'] = srs.GetAttrValue('SPHEROID')
    d['ellipsoid_ra'] = srs.GetSemiMajor()
    d['ellipsoid_reciprocal_flattening'] = srs.GetInvFlattening()
    # TODO only works for switzerland at the moment
    # Datum Information
    sr2 = _osr.SpatialReference()
    sr2.ImportFromEPSG(21781)
    datum = sr2.GetTOWGS84()
    d['datum_name'] = 'Swiss National 3PAR'
    d['datum_shift_dx'] = 679.396
    d['datum_shift_dy'] = -0.095
    d['datum_shift_dz'] = 406.471
    # Projection Information
    d['false_easting'] = srs.GetProjParm('false_easting')
    d['false_northing'] = srs.GetProjParm('false_northing')
    d['projection_k0'] = srs.GetProjParm('scale_factor')
    d['center_longitude'] = srs.GetProjParm('longitude_of_center')
    d['center_latitude'] = srs.GetProjParm('latitude_of_center')
    out_type = type_mapping[d['data_format']]
    write_dataset(DEM, d, par_path, bin_path)


def gpri_raw_strides(nsamp, nchan, npat, itemsize):
    """
    This function computes the array strides for 
    the gpri raw file format
    """
    # The first stride jumps from one record to the
    # next corresponding to the same pattern (AAA etc)
    st_az = ((nsamp + 1) * nchan) * npat * itemsize
    # The second stride, for the range samples
    # jumps from one range sample to the next, they
    # alternate between one channel and another
    st_rg = itemsize * nchan
    # The third stride, for the two receivers, from
    # one channel to the other
    st_chan = itemsize
    # The final stride is for the polarimetric channels,
    # they are in subsequent records
    st_pol = ((nsamp + 1) * nchan) * itemsize
    # The full strides
    return (st_az, st_rg, st_chan, st_pol)


def load_raw(par_path, path):
    par = par_to_dict(par_path)
    nsamp = par['CHP_num_samp']
    nchan = 2
    npat = len(par['TX_RX_SEQ'].split('-'))
    itemsize = _np.int16(1).itemsize
    bytes_per_record = (nsamp + 1) * 2 * itemsize
    filesize = _osp.getsize(path + '.raw')
    raw = _np.memmap(path + '.raw', dtype='int16', mode='r')
    nl_tot = int(filesize / bytes_per_record)
    sh = (nl_tot / npat, nsamp + 1, \
          nchan, npat)
    stride = gpri_raw_strides(nsamp, nchan, npat, itemsize)
    raw_shp = _ast(raw, shape=sh, strides=stride)
    return raw_shp

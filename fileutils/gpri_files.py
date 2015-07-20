# -*- coding: utf-8 -*-
"""
Created on Thu May 15 13:51:30 2014

@author: baffelli
"""
import re
import numpy as _np
import dateutil.parser
import string as _str
import other_files
import os.path as _osp
from numpy.lib.stride_tricks import as_strided as _ast
from osgeo import gdal as _gdal, osr as _osr
import mmap as _mm
import array as _arr

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


def par_to_dict(par_file):
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
    #Map type to gamma
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
        #We dont have a SAR image,
        #try as it were a DEM
        try:
                shape = (par_dict['nlines'],
                par_dict['width'])
        except:
        #Last try
        #interferogram
                try:
                    shape = (par_dict['interferogram_width'], par_dict['interferogram_azimuth_lines'])
                except:
                    raise KeyError("This file does not contain data shape specification in a known format")
    if memmap:
        with open(bin_file, 'rb') as mmp:
            buffer = _mm.mmap(mmp.fileno(), 0, prot=_mm.PROT_READ)
            d_image = _np.ndarray(shape, dt, buffer)
        #d_image = _np.memmap(bin_file, shape=shape, dtype=dt, mode='r')
    else:
        d_image = _np.fromfile(bin_file, dtype=dt).reshape(shape)
    return d_image, par_dict


def write_dataset(dataset, par_dict, par_file, bin_file):
    #Write the  binary file
    try:
        dataset.tofile(bin_file)
    except AttributeError:
        raise TypeError("The dataset is not a numpy ndarray")
    #Write the parameter file
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
    #FOrmat information
    if _np.issubdtype(DEM.dtype, _np.int32):
        DEM = DEM.astype(_np.int16)
        d['data_format'] = 'INTEGER*2'
    elif _np.issubdtype(DEM.dtype, _np.int16):
        d['data_format'] = 'INTEGER*2'
    elif _np.issubdtype(DEM.dtype, _np.float32):
        d['data_format'] = 'REAL*4'
    #Geotransform information
    d['DEM_scale'] = 1.0
    d['DEM_projection'] = 'OMCH'
    d['corner_north'] = GT[3]
    d['corner_east'] = GT[0]
    d['width'] = gt.RasterXSize
    d['nlines'] = gt.RasterYSize
    d['post_north'] = GT[5]
    d['post_east'] = GT[1]
    #Ellipsoid information
    d['ellipsoid_name'] = srs.GetAttrValue('SPHEROID')
    d['ellipsoid_ra'] = srs.GetSemiMajor()
    d['ellipsoid_reciprocal_flattening'] = srs.GetInvFlattening()
    #TODO only works for switzerland at the moment
    #Datum Information
    sr2 = _osr.SpatialReference()
    sr2.ImportFromEPSG(21781)
    datum=sr2.GetTOWGS84()
    d['datum_name'] = 'Swiss National 3PAR'
    d['datum_shift_dx'] = 679.396
    d['datum_shift_dy'] = -0.095
    d['datum_shift_dz'] = 406.471
    #Projection Information
    d['false_easting'] = srs.GetProjParm('false_easting')
    d['false_northing'] = srs.GetProjParm('false_northing')
    d['projection_k0'] = srs.GetProjParm('scale_factor')
    d['center_longitude'] = srs.GetProjParm('longitude_of_center')
    d['center_latitude'] = srs.GetProjParm('latitude_of_center')
    out_type = type_mapping[d['data_format']]
    write_dataset(DEM, d, par_path,  bin_path)





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


def load_raw(path):
    par = load_par(path + '.raw_par')
    nsamp = par['CHP_num_samp'][0]
    nchan = 2
    npat = len(par['TX_RX_SEQ'][0].split('-'))
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



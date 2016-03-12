# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:00:59 2014

@author: baffelli
"""
import numpy as _np
from collections import OrderedDict as _od
from osgeo import osr as _osr
from osgeo import gdal as _gdal

def load_dat(path):
    """
    Load xdr SAR image 

    Parameters 
    ----------
    path : string
    The path of the file to load    
    Returns
    -------
    data : ndarray
        the data as a numpy array
    """
    # sizes from file
    dt_head = _np.dtype('>i4')
    temp_path = path
    f = open(temp_path)
    dim_vec = _np.zeros(2,dtype = dt_head)
    dim_vec = _np.fromfile(f,dtype=dt_head,count = 2)
    #Compute sizes
    n_range = dim_vec[0]
    n_az = dim_vec[1]
    #Skip header
    f.seek(8, 0)  # seek
    #Define new data type
    dt_data = _np.dtype('>c8')
    data =  _np.fromfile(f,dtype=dt_data)
    data = data.reshape([n_az,n_range])
    return data
    
def load_geotiff(path):
    from osgeo import gdal
    ds = gdal.Open(path)
    return ds


def load_shapefile(path, mode=0):
    from osgeo import gdal, osr, ogr
    #Create driver
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(path, mode)
    if dataSource is None:
        print("Could not open " + str(path))
    else:
        return dataSource

#TODO make it work for other cuntries
def gdal_to_dict(ds):
    #Mapping from wkt to parameters
    """
     * [0]  Spheroid semi major axis
     * [1]  Spheroid semi minor axis
     * [2]  Reference Longitude
     * [3]  Reference Latitude
     * [4]  First Standard Parallel
     * [5]  Second Standard Parallel
     * [6]  False Easting
     * [7]  False Northing
     * [8]  Scale Factor
    """
    #Get projection information from wkt
    wkt = ds.GetProjection()
    srs = _osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    proj_arr = srs.ExportToPCI()
    #Array for the ellipsoid
    ell_arr = proj_arr[2]
    #Create new dict
    proj_dict = _od()
    #Part 1: General Parameters
    proj_dict['title'] = 'DEM'
    proj_dict['DEM_projection'] = 'OMCH'
    #Set the type according to the dem type
    tp = _gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
    print(tp)
    if tp == 'Float32':
        proj_dict['data_format'] = 'REAL*4'
    if tp == 'Int32':
        proj_dict['data_format'] = 'INTEGER*2'
    if tp == 'UInt16':
        proj_dict['data_format'] = 'SHORT INTEGER'
    proj_dict['DEM_hgt_offset'] = 0
    proj_dict['DEM_scale'] = 1.0
    proj_dict['width'] = ds.RasterXSize
    proj_dict['nlines'] = ds.RasterYSize
    gt = ds.GetGeoTransform()
    proj_dict['corner_east'] = [gt[0], 'm']
    proj_dict['corner_north'] = [gt[3], 'm']
    proj_dict['post_north'] = [gt[5], 'm']
    proj_dict['post_east'] = [gt[1], 'm']
    #TODO allow using other ellipsods
    #Part 2: Ellipsoid Parameters
    proj_dict['ellipsoid_name'] = 'Bessel 1841'
    proj_dict['ellipsoid_ra'] = [ell_arr[0], 'm']
    rf = ell_arr[0] / (ell_arr[0] - ell_arr[1])
    proj_dict['ellipsoid_reciprocal_flattening'] = rf
    #TODO allow using other datums
    #Part 3: Datum Parameters
    proj_dict['datum_name'] = 'SWiss National 3PAR'
    proj_dict['datum_shift_dx'] = [679.396, 'm']
    proj_dict['datum_shift_dy'] = [-0.095, 'm']
    proj_dict['datum_shift_dz'] = [406.471, 'm']
    proj_dict['datum_scale_m'] = 0.0
    proj_dict['datum_rotation_alpha'] = [0.0, 'arc-sec']
    proj_dict['datum_rotation_beta'] = [0.0, 'arc-sec']
    proj_dict['datum_rotation_gamma'] = [0.0, 'arc-sec']
    #Part 4: Projection Parameters for UTM, TM, OMCH, LCC, PS, PC, AEAC, LCC2, OM, HOM coordinates
    proj_dict['projection_name'] = 'OM - Switzerland'
    if proj_dict['DEM_projection'] in ['UTM', "TM", "OMCH", "LCC", "PS", "PC", "AEAC", "LCC2", "OM", "HOM"]:
        proj_dict['center_latitude'] = ell_arr[2]
        proj_dict['center_longitude'] = ell_arr[3]
        proj_dict['projection_k0'] = ell_arr[8]
        proj_dict['false_easting'] = ell_arr[6]
        proj_dict['false_northing'] = ell_arr[7]

    return proj_dict

def new_to_old_gt(gt):
    """
    Convert a new style swiss gt (with sitxh digit coordinate)
    to the old style gt
    """
    gt_new = list(gt)
    gt_new[0] = gt[0] - 2e6
    gt_new[3] = gt[3] - 1e6
    return gt_new

def dict_to_gt(dic):
    from osgeo import gdal, osr
    
    
def save_dem(ds,path):
    """
    This function saves a geotiff file in the gamma format
    Parameters
    ----------
    ds : osgeo.gdal.Dataset
        A gdal dataset containing a DEM
    path : str
        The path to save the dem
    """
    from osgeo import gdal



def wkt_to_dict(wkt):
    """
    This function convers a WKT coordinate system definition
    into a dictionary whose keys correspond to gammas
    convention
    Parameters
    ----------
    wkt : str
        The wkt to convert
    Returns
    -------
    dict
    """
    """
     * [0]  Spheroid semi major axis
     * [1]  Spheroid semi minor axis
     * [2]  Reference Longitude
     * [3]  Reference Latitude
     * [4]  First Standard Parallel
     * [5]  Second Standard Parallel
     * [6]  False Easting
     * [7]  False Northing
     * [8]  Scale Factor
    """
    from osgeo import gdal, osr
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    proj_arr = srs.ExportToPCI()
    proj_dict = _od()
    #TODO fix dem projection to be flexible
    proj_dict['DEM_projection'] = 'OMCH'
    ell_arr = proj_arr[2]
    proj_dict['ellipsoid_ra'] = ell_arr[0]
    rf = ell_arr[0] / (ell_arr[0] - ell_arr[1])
    proj_dict['ellipsoid_reciprocal_flattening'] = rf
    proj_dict['center_latitude'] = ell_arr[2]
    proj_dict['center_longitude'] = ell_arr[3]
    proj_dict['projection_k0'] = ell_arr[8]
    proj_dict['false_easting'] = ell_arr[6]
    proj_dict['false_northing'] = ell_arr[7]
    return proj_dict

def load_bin(path,dim):
    """
    Load bin SAR image in the format saved by POLSARPro
    
    Parameters 
    ----------
    path : string
        The path of the file to load
    dim : iterable
        The size of the array to read the data into
    
    Returns
    -------
    read_data : ndarray
        the data as a numpy array
    """
    read_data = _np.fromfile(file=path, dtype=_np.complex64).reshape(dim)
    return read_data

def load_scattering(paths,fun=load_dat,other_args=[]):
    """
    Load scattering matrices from a list of paths
    
    Parameters
    ----------
    paths: iterable
        Iterable containing the list of paths to load from
    fun : optional, function
        The function needed to load the corresponding files
    other_args : optional, list
        A list of arguments to pass to the loading function
   
    Returns
    -------
    read_data : list
        the data as a list of numpy arrays
    """
    read_data = []
    for path in paths:
        temp_args = [path,] + other_args
        temp_data = fun(*temp_args)
        read_data = read_data + [temp_data,]
    return read_data

def load_coherency(path, dim):
    """
    Load coherency matrix from the format saved by polsarpro
    (single files for each channel)
    Parameters 
    ----------
    path : string 
        the folder containing the elements to be loaded
    dim : iterable
        the size of the file    
    Returns
    -------
    data : ndarray 
        the data as a list of numpy arrays
    """
    data_type = _np.float32
    data = _np.zeros( dim + [3,3], dtype = _np.complex64)
    #Load channel
    path_name = path  + "T11.bin"
    read_data = _np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    data[:,:,0,0] = read_data
    path_name = path  + "T22.bin"
    read_data = _np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    data[:,:,1,1] = read_data
    path_name = path  + "T33.bin"
    read_data = _np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    data[:,:,2,2] = read_data
    path_name = path  + "T12_real.bin"
    read_data_real = _np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    path_name = path  + "T12_imag.bin"
    read_data_imag = _np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    data[:,:,0,1] = read_data_real + 1j* read_data_imag
    data[:,:,1,0] = read_data_real - 1j* read_data_imag
    path_name = path  + "T13_real.bin"
    read_data_real = _np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    path_name = path  + "T13_imag.bin"
    read_data_imag = _np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    data[:,:,0,2] = read_data_real + 1j* read_data_imag
    data[:,:,2,0] = read_data_real - 1j* read_data_imag
    path_name = path  + "T23_real.bin"
    read_data_real = _np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    path_name = path  + "T23_imag.bin"
    read_data_imag = _np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    data[:,:,1,2] = read_data_real + 1j* read_data_imag
    data[:,:,2,1] = read_data_real - 1j* read_data_imag
    return data


def slice_from_file(file):
    """
    This function returns a list
    of slice from a file
    :param file:
    :return:
    """

# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:00:59 2014

@author: baffelli
"""
import numpy as np

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
    dt_head = np.dtype('>i4')
    temp_path = path
    f = open(temp_path)
    dim_vec = np.zeros(2,dtype = dt_head)
    dim_vec = np.fromfile(f,dtype=dt_head,count = 2)
    #Compute sizes
    n_range = dim_vec[0]
    n_az = dim_vec[1]
    #Skip header
    f.seek(8, 0)  # seek
    #Define new data type
    dt_data = np.dtype('>c8')
    data =  np.fromfile(f,dtype=dt_data)
    data = data.reshape([n_az,n_range])
    return data
    
def load_geotiff(path):
    from osgeo import gdal
    ds = gdal.Open(path)
    return ds
    
def gdal_to_dict(ds):
    wkt = ds.GetProjection()
    d = wkt_to_dict(wkt)
    d['width'] = ds.RasterXSize
    d['nlines'] = ds.RasterYSize
    gt = ds.GetGeoTransform()
    d['corner_east'] = gt[0]
    d['corner_north'] = gt[3]
    d['post_north'] = gt[5]
    d['post_east'] = gt[1]
    return d

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
    proj_dict = dict()
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
    read_data = np.fromfile(file=path, dtype=np.complex64).reshape(dim)
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
    data_type = np.float32
    data = np.zeros( dim + [3,3], dtype = np.complex64)
    #Load channel
    path_name = path  + "T11.bin"
    read_data = np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    data[:,:,0,0] = read_data
    path_name = path  + "T22.bin"
    read_data = np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    data[:,:,1,1] = read_data
    path_name = path  + "T33.bin"
    read_data = np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    data[:,:,2,2] = read_data
    path_name = path  + "T12_real.bin"
    read_data_real = np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    path_name = path  + "T12_imag.bin"
    read_data_imag = np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    data[:,:,0,1] = read_data_real + 1j* read_data_imag
    data[:,:,1,0] = read_data_real - 1j* read_data_imag
    path_name = path  + "T13_real.bin"
    read_data_real = np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    path_name = path  + "T13_imag.bin"
    read_data_imag = np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    data[:,:,0,2] = read_data_real + 1j* read_data_imag
    data[:,:,2,0] = read_data_real - 1j* read_data_imag
    path_name = path  + "T23_real.bin"
    read_data_real = np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    path_name = path  + "T23_imag.bin"
    read_data_imag = np.fromfile(file=path_name, dtype=data_type).reshape(dim)
    data[:,:,1,2] = read_data_real + 1j* read_data_imag
    data[:,:,2,1] = read_data_real - 1j* read_data_imag
    return data

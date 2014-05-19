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

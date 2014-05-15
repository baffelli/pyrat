# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:00:59 2014

@author: baffelli
"""
import numpy as np

def load_dat(path):
    """
    Load xdr SAR image 
    --------
    Parameters 
    path: The path of the file to load
    --------
    Returns
    data: the data as a numpy array
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
    --------
    Parameters 
    path: The path of the file to load
    dim: The size of the array to read the data into
    --------
    Returns
    read_data: the data as a numpy array
    """
    read_data = np.fromfile(file=path, dtype=np.complex64).reshape(dim)
    return read_data

def load_scattering(paths,fun=load_dat,other_args=[]):
    """
    Load scattering matrices from a list of paths
    --------
    Parameters 
    paths: Iterable containing the list of paths to load from
    fun: The function needed to load the corresponding files
    other_args: A list of arguments to pass to the loading function
    --------
    Returns
    read_data: the data as a list of numpy arrays
    """
    read_data = []
    for path in paths:
        temp_args = [path,] + other_args
        temp_data = fun(*temp_args)
        read_data = read_data + [temp_data,]
    return read_data
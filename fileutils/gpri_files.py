# -*- coding: utf-8 -*-
"""
Created on Thu May 15 13:51:30 2014

@author: baffelli
"""
import re
import numpy as np
import dateutil.parser
import string
import other_files

def load_par(path):
    """
    Load Gamma Format .par parameter file in a dictionary
    --------
    Parameters 
    path: The path of the file to load
    --------
    Returns
    par: the dictionary of the parameters
    """
    par = []
    with open(path, 'r') as fin:
        lines=fin.readlines()
    for l in lines:
        keys = l.split()
        if len(keys) > 1:
            if keys[0] == "title:":
                try:
                    utc = dateutil.parser.parse(string.join(keys[1:3]))
                    par = par + [('utc',utc)]
                    continue
                except:
                    continue
            else:
                par_name = keys[0]
                par_name = par_name.replace(":","")
                par_numbers = re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", l)
                new_numbers  = []
                for num in par_numbers:
                    flt = float(num)
                    new_numbers  = new_numbers + [flt]
                par = par + [(par_name,new_numbers)]
    return dict(par)

def dict_to_par(par_dict):
    """
    This function converts a dict in  the gamma format parameter list
    Parameters
    ----------
    par_dict : dict
        A dictionary
    Returns
    -------
    str
    """
    a = ""
    for key, value in par_dict.iteritems():
        temp_str = str(key) + ':\t' + str(value) + '\n'
        a = a + temp_str
    return a
    
def geotif_to_dem(gt, path):
    """
    This function converts a geotiff 
    DEM into a gamma format pair
    of binary DEM and parameter file
    """
    DEM = gt.ReadAsArray()
    d = other_files.gdal_to_dict(gt)
    if np.issubdtype(DEM.dtype, np.int32):
        DEM = DEM.astype(np.int16)
        d['data_format'] = 'INTEGER*2'
    elif np.issubdtype(DEM.dtype, np.int16):
        d['data_format'] = 'INTEGER*2'
    elif np.issubdtype(DEM.dtype, np.float32):
        d['data_format'] = 'REAL*4'
    d['DEM_scale'] = 1.0
    d['projection_name'] = 'OMCH'
    DEM_par = dict_to_par(d)
    DEM = DEM.flatten()
    DEM.byteswap().tofile(path + '.dem')
    fi = open(path + '.dem.par', "w")
    fi.write(DEM_par)
    fi.close()
    

def load_complex(path):
    d = np.fromfile(file=path, dtype=np.float32)
    d = d.byteswap()
    d_real = d[0::2]
    d_imag = d[1::2]
    d_comp = d_real + 1j * d_imag
    return d_comp

def load_slc(path):
    """
    Load Gamma Format .slc image 
    --------
    Parameters 
    path: The path of the file to load
    --------
    Returns
    par: the file as a numpy array. The shape is (n_azimuth, n_range)
    """
    par = load_par(path +'.par')
    d_comp = load_complex(path)
    d_image = np.reshape(d_comp,(par['azimuth_lines'][0],par['range_samples'][0]))
    return d_image
    
def load_int(path):
    split_string = path.split('.')
    print split_string
    par_path = split_string[0] + '.off'
    print par_path
    par = load_par(par_path)
    d_comp = load_complex(path)
    d_image = np.reshape(d_comp,(par['interferogram_azimuth_lines'][0],par['interferogram_width'][0]))
    return d_image



def load_dem(path):
    split_string = path.split('.')
    print split_string
    par_path = split_string[0] + '.par'
    print par_path
    par = load_par(par_path)
    d = np.fromfile(file=path, dtype=np.float32).byteswap()
    d_image = np.reshape(d,(par['interferogram_azimuth_lines'][0],par['interferogram_width'][0]))
    return d_image
    



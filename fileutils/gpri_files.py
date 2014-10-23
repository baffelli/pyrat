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
                    utc = dateutil.parser.parse(_str.join(keys[1:3]))
                    par = par + [('utc',utc)]
                    continue
                except:
                    continue
            elif keys[0] == "image_format:":
                par = par + [('image_format', keys[1::])]
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
    if _np.issubdtype(DEM.dtype, _np.int32):
        DEM = DEM.astype(_np.int16)
        d['data_format'] = 'INTEGER*2'
    elif _np.issubdtype(DEM.dtype, _np.int16):
        d['data_format'] = 'INTEGER*2'
    elif _np.issubdtype(DEM.dtype, _np.float32):
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
    d = _np.fromfile(file=path, dtype=_np.float32)
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
    shape = (par['azimuth_lines'][0],par['range_samples'][0])
    if par['image_format'][0] == 'FCOMPLEX':
        dt = _np.dtype('complex64')
    elif par['image_format'][0] == 'SCOMPLEX':
        dt = _np.dtype('complex32')
    d_image = _np.memmap(path, shape = shape, dtype = dt).byteswap()
    return d_image
    
def load_int(path):
    split__str = path.split('.')
    print split__str
    par_path = split__str[0] + '.off'
    print par_path
    par = load_par(par_path)
    d_comp = load_complex(path)
    d_image = _np.reshape(d_comp,(par['interferogram_azimuth_lines'][0],par['interferogram_width'][0]))
    return d_image



def load_dem(path):
    split__str = path.split('.')
    print split__str
    par_path = split__str[0] + '.par'
    print par_path
    par = load_par(par_path)
    d = _np.fromfile(file=path, dtype=_np.float32).byteswap()
    d_image = _np.reshape(d,(par['interferogram_azimuth_lines'][0],par['interferogram_width'][0]))
    return d_image
    



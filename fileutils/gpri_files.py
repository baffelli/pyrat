# -*- coding: utf-8 -*-
"""
Created on Thu May 15 13:51:30 2014

@author: baffelli
"""
import re
import numpy as np
import dateutil.parser
import string

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
    par_path = split_string[0] + '.off'
    print par_path
    par = load_par(par_path)
    d = np.fromfile(file=path, dtype=np.float32).byteswap()
    d_image = np.reshape(d,(par['interferogram_azimuth_lines'][0],par['interferogram_width'][0]))
    return d_image
    



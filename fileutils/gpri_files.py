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



type_mapping ={
    'FCOMPLEX':_np.dtype('>c8'),
    'SCOMPLEX':_np.dtype('>c4'),
    'FLOAT':_np.dtype('>f4'),
    'SHORT INTEGER':_np.dtype('>i2')
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
            if len(l) > 1:
                par_dict[key] = l
            else:
                par_dict[key] = l[0]
    return par_dict


def dict_to_par(par_dict, par_file):
    with open(par_file,'w') as fout:
        for key,par in par_dict.iteritems():
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
    dt = type_mapping[par_dict['image_format']]
    shape = [par_dict['range_samples'],
             par_dict['azimuth_lines']]
    print(shape)
    if memmap:
        d_image = _np.memmap(bin_file, shape=shape, dtype=dt, mode='r')
    else:
        d_image = _np.fromfile(bin_file, dtype=dt).reshape(shape)
    return d_image, par_dict



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
            elif keys[0] == "TX_RX_SEQ:":
                par = par + [('TX_RX_SEQ', keys[1::])]
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
    
def path_helper(location, date, time, slc_dir = 'slc', data_dir= '/media/bup/Data'):
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
    def_path =  base_folder + slc_dir + '/' +  name
    return def_path

    
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
    
def save_complex(arr, path):
    with open(path,'wb') as out:
        out.write(arr.astype('>c8'))

def load_slc(path, memmap = True, sl = None):
    """
    Load Gamma Format .slc image 

    Parameters
    ----------
    path : string
    The path of the file to load
    memmap : boolean
    If set to true, load as a memmap
    sl : iterable
    List of slice to access only a part of the data
    Returns
    --------
        par: 
            the file as a numpy array. The shape is (n_azimuth, n_range)
    """
    par = load_par(path +'.par')
    shape = (par['azimuth_lines'][0],par['range_samples'][0])
    if par['image_format'][0] == 'FCOMPLEX':
        dt = _np.dtype('complex64')
    elif par['image_format'][0] == 'SCOMPLEX':
        dt = _np.dtype('complex32')
#    with open(path, 'rb') as slcfile:
#        if sl is None:
#            pass
#        else:
#            #We have to skip to the beginning of
#            #the block we are interested in
#            seeksize = _np.ravel_multi_index((el.start for el in sl),
#                                             shape)
#            slcfile.seek(0, seeksize)
#            
#            
    if memmap:
        d_image = _np.memmap(path, shape=shape, dtype=dt, mode='r').byteswap()
    else:
        d_image = _np.fromfile(path, dtype=dt)
    return d_image
    

    
def gpri_raw_strides(nsamp,nchan,npat, itemsize):
    """
    This function computes the array strides for 
    the gpri raw file format
    """
    #The first stride jumps from one record to the 
    #next corresponding to the same pattern (AAA etc)
    st_az = ((nsamp + 1) * nchan) * npat * itemsize
    #The second stride, for the range samples
    #jumps from one range sample to the next, they 
    #alternate between one channel and another
    st_rg = itemsize * nchan
    #The third stride, for the two receivers, from
    #one channel to the other
    st_chan =  itemsize
    #The final stride is for the polarimetric channels,
    #they are in subsequent records
    st_pol = ((nsamp + 1) * nchan)  * itemsize
    #The full strides
    return (st_az, st_rg, st_chan, st_pol) 
    
def load_raw(path):
    par = load_par(path +'.raw_par')
    nsamp = par['CHP_num_samp'][0]
    nchan = 2
    npat = len(par['TX_RX_SEQ'][0].split('-'))
    itemsize = _np.int16(1).itemsize
    bytes_per_record = (nsamp + 1) * 2 * itemsize
    filesize = _osp.getsize(path + '.raw')
    raw = _np.memmap(path + '.raw', dtype='int16', mode='r')
    nl_tot = int(filesize/bytes_per_record)
    sh = (nl_tot / npat, nsamp + 1,\
        nchan, npat)
    stride = gpri_raw_strides(nsamp, nchan, npat, itemsize)
    raw_shp = _ast(raw, shape=sh, strides=stride)
    return raw_shp

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
    



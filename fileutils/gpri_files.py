# -*- coding: utf-8 -*-
"""
Created on Thu May 15 13:51:30 2014

This module contains classes and function to deal with Gamma file formats


@author: baffelli
"""
import os.path as _osp
import os as _os
from osgeo import osr as _osr
import mmap as _mm
import copy as _cp
import numbers as _num
import tempfile as _tf
import scipy.signal as _sig

import numpy as _np
from numpy.lib.stride_tricks import as_strided as _ast
import scipy as _sp
from collections import namedtuple as _nt
from collections import OrderedDict as _od


#Constants
ra = 6378137.0000    #WGS-84 semi-major axis
rb = 6356752.3141    #WGS-84 semi-minor axis

C = 299792458.0    #speed of light m/s
KU_WIDTH = 15.798e-3 #WG-62 Ku-Band waveguide width dimension
KU_DZ = 10.682e-3   #Ku-Band Waveguide slot spacing
RANGE_OFFSET= 3



# This dict defines the mapping
# between the gamma datasets and numpy
type_mapping = {
    'FCOMPLEX': _np.dtype('>c8'),
    #'SCOMPLEX': _np.dtype('>c4'),
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

def get_image_size(path, width, type_name):
    """
    This function gets the shape of an image given the witdh
    and the datatype
    :param path:
    :param width:
    :param type:
    :return:
    """
    import os as _os
    fc = _os.path.getsize(path) / type_mapping[type_name].itemsize
    shape = [width, int(fc / (width))]
    return shape

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
            # Tuple of slices
            elif hasattr(sl, '__contains__'):
                # By taking the first element, we automatically have
                # the correct data
                try:
                    az_vec_sl = az_vec[sl[1]]
                    if hasattr(az_vec_sl, '__contains__'):
                        if len(az_vec_sl) > 1:
                            az_spac = az_vec_sl[1] - az_vec_sl[0]
                        else:
                            az_spac = az_spac
                        az_0 = az_vec_sl[0]
                    else:
                        az_0 = az_vec_sl
                        az_spac = self.GPRI_az_angle_step[0] * 1
                except:
                    pass
                try:
                    r_vec_sl = r_vec[sl[0]]
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
                except:
                    pass
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
        #fin.readline()
        for line in fin:
            if line:
                split_array = line.replace('\n','').split(':', 1)
                if len(split_array) > 1:
                    key = split_array[0]
                    if key == 'time_start':#The utc time string should not be split
                        l = split_array[1:]
                    else:
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
                        out = out + ' ' + str(p)
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
    #We start with the smallest stride, jumping
    #from rx channel to the other
    st_chan = itemsize
    #The second smallest jumps from one range sample to the next
    #so we have to jump to every nchan-th sample
    st_rg = st_chan * nchan
    #To move from one pattern to the next, we move to the next impulse
    st_pat = (nsamp +1) * st_rg
    #To move in azimuth to the subsequent record with the same
    #pattern , we have to jump npat times a range record
    st_az = st_pat * npat
    return (st_rg, st_az , st_chan, st_pat)


def load_raw(par_path, path):
    """
    This function loads a gamma raw dataset
    :param the path to the raw_par file:
    :param the path to the raw file:
    :return:
    """
    par = par_to_dict(par_path)
    nsamp = par['CHP_num_samp']
    nchan = 2
    npat = len(par['TX_RX_SEQ'].split('-'))
    itemsize = _np.int16(1).itemsize
    bytes_per_record = (nsamp + 1) * 2 * itemsize
    filesize = _osp.getsize(path)
    raw = _np.memmap(path, dtype='int16', mode='r')
    nl_tot = int(filesize / bytes_per_record)
    sh = (nsamp + 1, nl_tot / npat, \
          nchan, npat)
    stride = gpri_raw_strides(nsamp, nchan, npat, itemsize)
    raw_shp = _ast(raw, shape=sh, strides=stride)
    return raw_shp, par


def default_slc_dict():
    """
    This function creates a default dict for the slc parameters of the
    gpri, that the user can then fill according to needs
    :return:
    """
    par = _od()
    par['title'] = ''
    par['sensor'] = 'GPRI 2'
    par['date'] = [0, 0 ,0]
    par['start_time'] = 0
    par['center_time'] = 0
    par['end_time'] = 0
    par['azimuth_line_time'] = 0
    par['line_header_size'] = 0
    par['range_samples'] = 0
    par['azimuth_lines'] = 0
    par['range_looks'] = 1
    par['azimuth_looks'] = 1
    par['image_format'] = 'FCOMPLEX'
    par['image_geometry'] = 'SLANT_RANGE'
    par['range_scale_factor'] = 0
    par['azimuth_scale_factor'] = 0
    par['center_latitude'] = [0, 'degrees']
    par['center_longitude'] = [0, 'degrees']
    par['heading'] = [0, 'degrees']
    par['range_pixel_spacing'] = [0, 'm']
    par['azimuth_pixel_spacing'] = [0, 'm']
    par['near_range_slc'] = [0, 'm']
    par['center_range_slc'] = [0, 'm']
    par['far_range_slc'] = [0, 'm']
    par['first_slant_range_polynomial'] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    par['center_slant_range_polynomial'] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    par['last_slant_range_polynomial'] = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    par['incidence_angle'] = [0.0, 'degrees']
    par['azimuth_deskew'] = 'OFF'
    par['azimuth_angle'] = [0.0, 'degrees']
    par['radar_frequency'] = [0.0, 'Hz']
    par['adc_sampling_rate'] = [0.0, 'Hz']
    par['chirp_bandwidth'] = [0.0, 'Hz']
    par['prf'] = [0.0, 'Hz']
    par['azimuth_proc_bandwidth'] = [0.0, 'Hz']
    par['doppler_polynomial'] = [0.0, 0.0, 0.0, 0.0]
    par['doppler_poly_dot'] = [0.0, 0.0, 0.0, 0.0]
    par['doppler_poly_ddot'] = [0.0, 0.0, 0.0, 0.0]
    par['receiver_gain'] = [0.0, 'dB']
    par['calibration_gain'] = [0.0, 'dB']
    par['sar_to_earth_center'] = [0.0, 'm']
    par['earth_radius_below_sensor'] = [0.0, 'm']
    par['earth_semi_major_axis'] = [ra, 'm']
    par['earth_semi_minor_axis'] = [rb, 'm']
    par['number_of_state_vectors'] = 0
    par['GPRI_TX_mode']= ''
    par['GPRI_TX_antenna'] = ''
    par['GPRI_RX_antenna'] = ''
    par['GPRI_az_start_angle'] = [0, 'degrees']
    par['GPRI_az_angle_step'] = [0, 'degrees']
    par['GPRI_ant_elev_angle'] = [0, 'degrees']
    par['GPRI_ref_north'] = 0
    par['GPRI_ref_east'] = 0
    par['GPRI_ref_alt'] = [0, 'm']
    par['GPRI_geoid'] = [0, 'm']
    par['GPRI_scan_heading'] = [0, 'degrees']
    return par

class rawParameters:
    """
    This class computes several raw parameters from
    a raw_par file
    """
    def __init__(self, raw_dict, raw):
        self.grp = _nt('GenericDict', raw_dict.keys())(**raw_dict)
        self.nsamp = self.grp.CHP_num_samp
        self.block_length = self.nsamp + 1
        self.chirp_duration = self.block_length/self.grp.ADC_sample_rate
        self.pn1 = _np.arange(self.nsamp/2 + 1) 		#list of slant range pixel numbers
        self.rps = (self.grp.ADC_sample_rate/self.nsamp*C/2.)/self.grp.RF_chirp_rate #range pixel spacing
        self.slr = (self.pn1 * self.grp.ADC_sample_rate/self.nsamp*C/2.)/self.grp.RF_chirp_rate  + RANGE_OFFSET  #slant range for each sample
        self.scale = (abs(self.slr)/self.slr[self.nsamp/8])**1.5     #cubic range weighting in power
        self.ns_max = int(round(0.90 * self.nsamp/2))	#default maximum number of range samples for this chirp
        self.tcycle = (self.block_length)/self.grp.ADC_sample_rate    #time/cycle
        self.dt = type_mapping['SHORT INTEGER']
        self.sizeof_data = _np.dtype(_np.int16).itemsize
        self.bytes_per_record = self.sizeof_data * self.block_length  #number of bytes per echo
        #Get file size
        self.filesize = _osp.getsize(raw)
        #Number of lines
        self.nl_tot = int(self.filesize/self.bytes_per_record)
        #Stuff for angle
        if self.grp.STP_antenna_end != self.grp.STP_antenna_start:
            self.ang_acc = self.grp.TSC_acc_ramp_angle
            rate_max = self.grp.TSC_rotation_speed
            self.t_acc = self.grp.TSC_acc_ramp_time
            self.ang_per_tcycle = self.tcycle * self.grp.TSC_rotation_speed	#angular sweep/transmit cycle
        else:
            self.t_acc = 0.0
            self.ang_acc = 0.0
            rate_max = 0.0
            self.ang_per_tcycle = 0.0
        if self.grp.ADC_capture_time == 0.0:
            angc = abs(self.grp.antenna_end - self.grp.antenna_start) - 2 * self.ang_acc	#angular sweep of constant velocity
            tc = abs(angc/rate_max)			#duration of constant motion
            self.grp.capture_time = 2 * self.t_acc + tc 	#total time for scanner to complete scan
       #Frequenct vector
        self.freq_vec = self.grp.RF_freq_min + _np.arange(self.grp.CHP_num_samp, dtype=float) * self.grp.RF_chirp_rate/self.grp.ADC_sample_rate

    def compute_slc_parameters(self, args):
        self.rmax = self.ns_max * self.rps;		#default maximum slant range
        self.win =  _sig.kaiser(self.nsamp, args.kbeta)
        self.zero = args.zero
        self.win2 = _sig.hanning(2*self.zero)		#window to remove transient at start of echo due to sawtooth frequency sweep
        self.ns_min = int(round(args.rmin/self.rps))	#round to the nearest range sample
        self.ns_out = (self.ns_max - self.ns_min) + 1
        self.rmin = self.ns_min * self.rps
        self.dec = args.d
        self.nl_acc = int(self.t_acc/(self.tcycle*self.dec))
        self.nl_tot = int(self.grp.ADC_capture_time/(self.tcycle))
        self.nl_tot_dec = self.nl_tot / self.dec
        self.nl_image = self.nl_tot_dec - 2 * self.nl_acc
        self.image_time = (self.nl_image - 1) * (self.tcycle * self.dec)
        if(args.rmax != 0.0):	#check if greater than maximum value for the selected chirp
          if (int(round(args.rmax/self.rps)) <= self.ns_max):
            self.ns_max = int(round(args.rmax/self.rps))
            self.rmax = self.ns_max * self.rps;
          else:
            print 'ERROR: requested maximum slant range exceeds maximum possible value with this chirp: %.3f'%(self.rmax,)

        self.ns_out = (self.ns_max - self.ns_min) + 1	#number of output samples
        #Compute antenna positions
        if self.grp.STP_antenna_end > self.grp.STP_antenna_start: #clockwise rotation looking down the rotation axis
            self.az_start = self.grp.STP_antenna_start + self.ang_acc	#offset to start of constant motion + center-freq squint
        else:				#counter-clockwise
            self.az_start = self.grp.STP_antenna_start - self.ang_acc

class rawData(_np.ndarray):


    def __array_finalize__(self, obj):
        if obj is None: return
        if hasattr(obj, '__dict__'):
            self.__dict__ = _cp.deepcopy(obj.__dict__)

    def __new__(cls, *args, **kwargs):
        data, par_dict = load_raw(args[0], args[1])
        obj = data.view(cls)
        obj.__dict__ = _cp.deepcopy(par_dict)
        #Duration of chirp
        obj.tcycle = obj.shape[0] * 1/obj.ADC_sample_rate
        return obj

    def channel_index(self, pat, ant):
        chan_list = self.TX_RX_SEQ.split('-')
        chan_idx = chan_list.index(pat)
        ant_map = {'u':0,'l':1}
        return [ant_map[ant], chan_idx]


    def azspacing(self):
        npat = len(self.TX_RX_SEQ.split('-'))
        return self.tcycle * self.STP_rotation_speed * npat

    def freqvec(self):
        return self.RF_freq_min + _np.arange(self.CHP_num_samp, dtype=float) * self.RF_chirp_rate/self.ADC_sample_rate

    az_spacing = property(azspacing)
    freq_vec = property(freqvec)





def lamg(freq, w):
    """
    This function computes the wavelength in waveguide for the TE10 mode
    """
    la = lam(freq)
    return la / _np.sqrt(1.0 - (la / (2 * w))**2)	#wavelength in WG-62 waveguide

#lambda in freespace
def lam(freq):
    """
    This function computes the wavelength in freespace
    """
    return C/freq

def squint_angle(freq, w, s):
    """
    This function computes the direction of the main lobe of a slotted
    waveguide antenna as a function of the frequency, the size and the slot spacing.
    It supposes a waveguide for the TE10 mode
    """
    sq_ang = _np.arccos(lam(freq) / lamg(freq, w) -  lam(freq) / (2 * s))
    dphi = _np.pi *(2.*s/lamg(freq, w) - 1.0)				#antenna phase taper to generate squint
    sq_ang_1 = _np.rad2deg(_np.arcsin(lam(freq) *dphi /(2.*_np.pi*s)))	#azimuth beam squint angle
    return sq_ang_1


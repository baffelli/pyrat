# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:32:36 2014

@author: baffelli
"""
import numpy as _np
import matplotlib as _mpl
import matplotlib.pyplot as _plt
import pyrat
import pyrat.core.polfun
import cv2 as _cv2
import scipy.fftpack as _fftp
import scipy.ndimage as _ndim

#Define colormaps
def psi_cmap(bounds):
    """
    This function computes
    a colormap to display 
    displacements given a series of boundaries
    Parameters
    ----------
    bounds : iterable
        A list of bounds
    Returns
    -------
    cm : matplotlib.colors.Colormap
        The colormap
    norm : matplotlib.colors.BoundaryNorm
        The boundary norm used to display
        the image with the colormap
    """
    color_list = (
    (75,185,79),
    (120, 196, 78),
    (177, 213, 63),
    (244, 238, 15),
    (253, 186, 47),
    (236, 85, 42),
    (222, 30, 61)
    )
    color_list = _np.array(color_list)/255.0
    cm = _mpl.colors.ListedColormap(color_list, name='defo')
    norm = _mpl.colors.BoundaryNorm(bounds, cm.N)
    return cm, norm
    
    
def draw_shapefile(path, basename="VEC200"):
    from osgeo import gdal, osr, ogr
    #Create driver
    layers = [    'Building', 
                  'FlowingWater',
                  'LandCover',
                  'Lake',
                  'Road']
    for l in layers:
        layer_name = path + basename + '_' + l + '.shp'
        dataSource = pyrat.other_files.load_shapefile(layer_name)

def compute_dim(WIDTH,FACTOR):
    """
    This function computes the figure size
    given the latex column width and the desired factor
    """
    fig_width_pt  = WIDTH * FACTOR
    
    inches_per_pt = 1.0 / 72.27
    golden_ratio  = (_np.sqrt(5) - 1.0) / 2.0  # because it looks good
    
    fig_width_in  = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio   # figure height in inches
    fig_dims      = [fig_width_in, fig_height_in] # fig dims as a list
    return fig_dims
    

def set_figure_size(f, x_size, ratio):
    y_size = x_size * ratio
    f.set_size_inches(x_size, y_size, forward=True)

def scale_array(*args,**kwargs):
    """
    This function scales an array between 0 and 1.
    
    Parameters
    ----------
    data : ndarray
        The array to be scaled.
    min_val : double
        The minimum value at which to cut the data.
    max_val : double
        The maximum value at which to clip the data.
    Returns
    -------
    ndarray
        The rescaled array.
    """
    data = args[0]
    if 'min_val' in kwargs:
        minVal = kwargs.get('min_val')
    else:
        minVal = _np.nanmin(data)
    if 'max_val' in kwargs:
        maxVal = kwargs.get('max_val')
    else:
        maxVal = _np.nanmax(data)
    scaled = (_np.clip(data,minVal,maxVal) - minVal) / (maxVal - minVal)
    return scaled

def segment_image(S, thresh):
    return (S>thresh).astype(_np.bool)

def copy_and_modify_gt(RAS,gt):
    from osgeo import gdal
    mem_drv = gdal.GetDriverByName( 'MEM' )
    output_dataset = mem_drv.Create('', RAS.shape[1], RAS.shape[0], RAS.shape[2], gdal.GDT_Float32)  
    output_dataset.SetProjection(gt.GetProjection())  
    output_dataset.SetGeoTransform(gt.GetGeoTransform())  
    for n_band in range(RAS.shape[-1] - 1):
        output_dataset.GetRasterBand(n_band + 1).WriteArray( RAS[:,:,n_band] )
    return output_dataset

def raster_to_geotiff(raster, ref_gt, path):
    """
    This function saves any ndarray
    as a geotiff with the geotransfrom
    take fron the reference gdal object specified as
    Parameters
    ----------
    raster : ndarray
        The object to save
    ref_gt : osgeo.gdal.Dataset
        The reference object
    path : string
        The path to save the geotiff to
    """
    #Remove nans
    from osgeo import gdal
    driver = gdal.GetDriverByName('GTiff')
    if raster.ndim is 3:
        nchan = raster.shape[2]
    else:
        nchan = 1
    dataset = driver.Create(
        path,
        raster.shape[1],
        raster.shape[0],
        nchan,
        numeric_dt_to_gdal_dt(ref_gt.GetRasterBand(1).DataType))
    dataset.SetGeoTransform(ref_gt.GetGeoTransform())
    dataset.SetProjection(ref_gt.GetProjection())
    if nchan > 1:
        for n_band in range(nchan -  1):
            band = raster[:,:,n_band]
            dataset.GetRasterBand(n_band + 1).WriteArray(band )
            dataset.FlushCache()  # Write to disk.
    else:
         dataset.GetRasterBand(1).WriteArray(raster )
         dataset.FlushCache()  # Write to disk.
#    dataset.GDALClose()
    return dataset
    
def upgrade_CH1903_gt(ds):
    """
    This function corrects the geotransform
    form the CH1903+ data which do not have the coordinate 
    in the new system (using 6 instead of 5 digits)
    CAUTION: no check is made that the previous geotransform is in
    the old system
    """
    from osgeo import gdal
    GT = ds.GetGeoTransform()
    GT_new = list(GT)
    GT_new[0] = GT[0] + 2e6
    GT_new[3] = GT[3] + 1e6
    mem_drv = gdal.GetDriverByName( 'MEM' )
    ds_copy = mem_drv.CreateCopy('',ds)
    ds_copy.SetGeoTransform(GT_new)
    return ds_copy

    
    
def numeric_dt_to_gdal_dt(number):
    from osgeo import gdal
    conversion_dict = {\
    1 : gdal.GDT_Byte,
    2 : gdal.GDT_UInt16,
    3 : gdal.GDT_Int16,
    4 : gdal.GDT_UInt32 ,
    5 : gdal.GDT_Int32,
    6 : gdal.GDT_Float32,
    }
    return conversion_dict[number]


def numpy_dt_to_numeric_dt(numpy_dt):
  NP2GDAL_CONVERSION = {
  "uint8": 1,
  "int8": 1,
  "uint16": 2,
  "int16": 3,
  "uint32": 4,
  "int32": 5,
  "float32": 6,
  "float64": 7,
  "complex64": 10,
  "complex128": 11,
  }
  return NP2GDAL_CONVERSION[numpy_dt]



def auto_azimuth(DOM, sec, tiepoint, initial_heading, center):
    def ftm(heading):
        DOM_sec = segment_GT(DOM, center, sec, heading)
        DOM_sec = resample_DEM(DOM_sec,[0.25, -0.25])
        DEM_seg, lut_sec, rev_lut_sec, xrad, yrad, ia, area, area_beta, r_sl = gc_map(DOM_sec, center, 
                                                                        sec, heading, seg_DEM =False)
        #Find maximum amplitude
        ampl = _np.abs(sec[:,:,0,0])
        max_idx = _np.nanargmax(ampl)
        max_pos = _np.unravel_index(max_idx, sec.shape[0:2] )
        #Convert position into coordinate
        max_pos_gc = rev_lut_sec[max_pos]
        max_pos_gc = (max_pos_gc.real, max_pos_gc.imag)
        max_coord = raster_index_to_coordinate(DOM_sec, max_pos_gc)
        err =  _np.array(tiepoint[0:2]) - _np.array(max_coord)
        return _np.linalg.norm(err)
    import scipy
    res = scipy.optimize.minimize_scalar(ftm, bounds = (0, 2 * _np.pi))
    return res
    
    
def reproject_gt(gt_to_project, gt_reference):

    """
    This function reporjects a gtiff
    to the projection specified by the other gtiff
    Parameters
    ----------
    gt_to_project : osgeo.gdal.Dataset
        The geotiff to project
    gt_reference : osgeo.gdal.Dataset
        The geotiff taken as a reference
    """
    from osgeo import gdal, osr
    tx = osr.CoordinateTransformation(osr.SpatialReference(gt_to_project.GetProjection()),\
    osr.SpatialReference(gt_reference.GetProjection()))
    geo_t = gt_to_project.GetGeoTransform()
    geo_t_ref = gt_reference.GetGeoTransform()
    x_size = gt_reference.RasterXSize # Raster xsize
    y_size = gt_reference.RasterYSize # Raster ysize
    (ulx, uly, ulz ) = tx.TransformPoint( geo_t[0], geo_t[3])
    (lrx, lry, lrz ) = tx.TransformPoint( geo_t[0] + geo_t_ref[1]*x_size, \
                                          geo_t[3] + geo_t_ref[5]*y_size )
     #Compute new geotransform
    new_geo = ( geo_t_ref[0], geo_t_ref[1], geo_t[2], \
                geo_t_ref[3], geo_t_ref[4], geo_t_ref[5] )
    mem_drv = gdal.GetDriverByName( 'MEM' )
    pixel_spacing_x = geo_t_ref[1]
    pixel_spacing_y = geo_t_ref[5]
    dest = mem_drv.Create('', int((lrx - ulx)/_np.abs(pixel_spacing_x)), \
            int((uly - lry)/_np.abs(pixel_spacing_y)), gt_to_project.RasterCount,\
            numeric_dt_to_gdal_dt(gt_to_project.GetRasterBand(1).DataType))
    dest.SetGeoTransform(new_geo)
    dest.SetProjection(gt_reference.GetProjection())
    res = gdal.ReprojectImage( gt_to_project, dest, \
                gt_to_project.GetProjection(), gt_reference.GetProjection(), \
                gdal.GRA_Bilinear )
    return dest
    

def reproject_radar(S, S_ref):
    """
    This function reprojects
    a radar image
    to the sample range and azimuth spacing
    of a given image and coregisters them
    """
    #Azimuth and range spacings
    az_sp = S.az_vec[1] - S.az_vec[0]  
    az_sp_ref = S_ref.az_vec[1] - S_ref.az_vec[0]
    rg_sp = S.r_vec[1] - S.r_vec[0]  
    rg_sp_ref = S_ref.r_vec[1] - S_ref.r_vec[0]
    #Compute new azimuth and range vectors
    az_vec_new = S_ref.az_vec
    rg_vec_new = S_ref.r_vec
    az_idx = _np.arange(S_ref.shape[0]) * az_sp_ref / az_sp
    rg_idx = _np.arange(S_ref.shape[1]) * rg_sp_ref / rg_sp
    az, r = _np.meshgrid(az_idx, rg_idx, order = 'ij')
    int_arr = bilinear_interpolate(S, r.T, az.T).astype(_np.complex64)
    S_res = S.__array_wrap__(int_arr)
    S_res.az_vec = az_vec_new
    S_res.r_vec = rg_vec_new
    return S_res
    
    
def coarse_coregistration(master, slave, sl):
    #Determine coarse shift
    T = _np.abs(master[ sl + (0, 0) ]).astype(_np.float32)
    I = _np.abs(slave[:,:,0,0]).astype(_np.float32)
    T[_np.isnan(T)] = 0
    I[_np.isnan(I)] = 0
    res = _cv2.matchTemplate(I,T, _cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = _cv2.minMaxLoc(res)
    sh = (sl[0].start - max_loc[1], sl[1].start - max_loc[0])
    print sh
    slave_coarse = shift_image(slave, (-sh[0], -sh[1]))
    return slave_coarse, res
    
    
    
def correct_shift_radar_coordinates(slave, master, axes = (0,1), oversampling = (5,2), sl = None):
    import pyrat.gpri_utils.calibration as calibration
    master_1 = master.__copy__()
    if sl is None:
        M = _np.abs(master_1[:,:,0,0]).astype(_np.float32)
        S = _np.abs(slave[:,:,0,0]).astype(_np.float32)
    else:
        M = _np.abs(master_1[sl + (0,0)]).astype(_np.float32)
        S = _np.abs(slave[sl + (0,0)]).astype(_np.float32)
    #Get shift
    co_sh, corr = calibration.get_shift(M,S, oversampling = oversampling, axes=(0,1))
    print(co_sh)
    slave_1 = shift_image(slave, co_sh)
    return slave_1, corr
    

def shift_image(image, shift):
    x = _np.arange(image.shape[0]) + shift[0] 
    y = _np.arange(image.shape[1]) + shift[1]
    x,y = _np.meshgrid(x, y, indexing='xy')
    image_1 = image.__array_wrap__(\
    bilinear_interpolate(_np.array(image), y.T, x.T))
    image_1[_np.isnan(image_1)] = 0
    return image_1
   
def shift_image_FT(image, shift):
    #Pad at least three times the shift required
    edge_pad_size = zip([0] * image.ndim,[0] * image.ndim)
    axes = range(len(shift))
    for ax in axes:
        ps = abs(int(shift[ax])) * 0
        edge_pad_size[ax] = (ps, ps)
    image_pad = _np.pad(image, edge_pad_size, mode ='constant')
    #Transform in fourier domain
    image_hat = _fftp.fftn(image_pad, axes=axes)
##    #Generate frequency vectors
#    freqs = [_fftp.ifftshift(_fftp.fftfreq(sh)) for sh in image_hat.shape]
#    #Compute shift ramp
#    fr = _np.zeros(image_hat.shape)
#
#    for f, sh, d, ax in zip(freqs, shift,\
#    image_hat.shape, range(image_hat.ndim)):
#        #Slicing for broadcasting
#        sl_bc = [None] * image_hat.ndim
#        sl_bc[ax] = Ellipsis
#        fr = fr + (f[sl_bc] * sh) / _np.double(d) 
#    ramp = _np.exp(-1j * 2* _np.pi * fr)
#    #Apply fourier shift theorem
    image_hat_shift = _ndim.fourier_shift(image_hat, shift)
#    image_hat_shift = image_hat * ramp
    image_shift = _fftp.ifftn(image_hat_shift, axes=axes)
    return image_shift
    
def resample_image(image, sampling_factor):
    x = _np.linspace(0,image.shape[0], num = image.shape[0] * sampling_factor[0])
    y = _np.linspace(0,image.shape[1], num = image.shape[1] * sampling_factor[1])
    x,y = _np.meshgrid(x, y, order = 'xy')
    image_1 = image.__array_wrap__(\
    bilinear_interpolate(_np.array(image), y.T, x.T))
    image_1[_np.isnan(image_1)] = 0
    return image_1
    
def resample_DEM(DEM,new_posting):
    """
    This function reporjects a gtiff
    to a new posting
    
    Parameters
    ----------
    DEM : osgeo.gdal.Dataset
        The geotiff to project
    new_posting : iterable
        The desired new posting
    """
    from osgeo import gdal, osr
    geo_t = DEM.GetGeoTransform()
    
    x_size = DEM.RasterXSize * _np.int(_np.abs(geo_t[1]/new_posting[0])) # Raster xsize
    y_size = DEM.RasterYSize * _np.int(_np.abs(geo_t[5]/new_posting[1]))# Raster ysize

     #Compute new geotransform
    new_geo = ( geo_t[0], new_posting[0], geo_t[2], \
                geo_t[3], geo_t[4], new_posting[1] )
    mem_drv = gdal.GetDriverByName( 'MEM' )
    dest = mem_drv.Create('', x_size, y_size, DEM.RasterCount,\
            numeric_dt_to_gdal_dt(DEM.GetRasterBand(1).DataType))
    dest.SetGeoTransform(new_geo)
    dest.SetProjection(DEM.GetProjection())
    res = gdal.ReprojectImage( DEM, dest, \
                DEM.GetProjection(), DEM.GetProjection(), \
                gdal.GRA_Bilinear )
    return dest    

def write_gt(arr, GT, proj):
    """
    This function writes a _np array to a
    gdal geotiff object
    
    Parameters
    ----------
    arr : ndarray
        The array to write
    GT  : iterable
        The geotransform
    proj : osr.SpatialReference
        The projection of the geotiff
    """
    from osgeo import gdal, gdal_array, osr
    x_size = arr.shape[1]
    y_size = arr.shape[0]
    #Determine number of bands
    if arr.ndim > 2:
        nbands = arr.shape[2]
    else:
        nbands = 1
    #Create a memory driver
    mem_drv = gdal.GetDriverByName( 'MEM' )
    dest = mem_drv.Create('', x_size, \
            y_size, nbands,\
            numeric_dt_to_gdal_dt(numpy_dt_to_numeric_dt(arr.dtype.name)))
    #Set geotransform and projection
    dest.SetGeoTransform(GT)
    dest.SetProjection(proj)
    if nbands > 1:
        for band_number in range(nbands):
            dest.GetRasterBand(band_number + 1).WriteArray(arr[:,:,band_number])
    else:
        dest.GetRasterBand(1).WriteArray(arr)
    return dest
    
    
    
def paletted_to_rgb(gt):
    """
    This function converts a paletted
    geotiff to a RGB geotiff
    """
    from osgeo import gdal
    ct = gt.GetRasterBand(1).GetColorTable()
    ct.GetColorEntry(3)
    #Get raster
    RAS = gt.ReadAsArray()
    RGBA_ras = _np.zeros(RAS.shape + (4,0))
    #Create vector of values:
    palette_list = _np.unique(RAS)
    rgba = _np.zeros((RAS.max() + 1 , 4))

    for idx in palette_list:
        color = _np.array(ct.GetColorEntry(int(idx))) / 255.0
        rgba[idx,:] = color
    RGBA_ras = rgba[RAS]
    mem_drv = gdal.GetDriverByName( 'MEM' )
    output_dataset = mem_drv.Create('', RAS.shape[1], RAS.shape[0], 4, gdal.GDT_Float32)  
    output_dataset = copy_and_modify_gt(RGBA_ras,gt)
    return output_dataset
    
def histeq(im,nbr_bins=256):
    """
    This function performs histogram equalization on a ndarray.
    
    Parameters
    ----------
    data : ndarray
        The image to be equalized.
    nbr_bins : int
        The number of histogram bins.
    Returns
    -------
    ndarray
        The equalized image.
    """
    #get image histogram
    imhist,bins = _np.histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = cdf / cdf[-1] #normalize
    #use linear interpolation of cdf to find new pixel values
    im2 = _np.interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape)

def stretch_contrast(im, tv = 5, ma = 95):
    """
    This function performs contrast
    stretching on an image by clipping
    it at the specified minimum and maximum percentiles
    
    Parameters
    ----------
    im : array_like
        The image to stretch
    mi : float
        The low percentile
    ma  : float
        The upper percentile
    """
    bv = _np.percentile(im, mi)
    tv = _np.percentile(im, ma)
    im1 = _np.clip(im, bv, tv)
    return im1

def exp_im(im, k):
    """
    Converts an image to the 0-1 range using a negative
    exponential scaling
    
    Parameters
    ----------
    im : array_like
        The image to convert
    k  : double
        The scaling exponent
    Returns
    -------
    ndarray
        The scaled image.
    """
    return scale_array(1 - _np.exp(- k * _np.abs(im)))
    
    
def compute_map_extent(MAP, center, S, heading):
    """
    This function computes the extent
    that a gpri Image occupies
    on a given geotiff MAP
    
    Parameters
    ----------
    Map : osgeo.gdal.Dataset
        The gdal Dataset
    center  : iterable
        The center position, where the gpri is placed
    S   : pyrat.ScatteringMatrix
        The object of interest
    heading : float
        The heading on the map in degrees
    """
    ext = get_extent(MAP)
    #Compute the area covered by radar
    #in DEM coordinates
    r_vec = (S.r_vec.max(),S.r_vec.min())
    az_vec = _np.mod(S.az_vec - heading, _np.pi * 2)
    r, az = _np.meshgrid(r_vec,az_vec)
    x = r * _np.sin(az) + center[0]
    y = r * _np.cos(az) + center[1]
    #Compute limits of area covered and clip to the DEM size
    x_lim = _np.clip(_np.sort((x.min(),x.max())),_np.min(ext[0]),_np.max(ext[0]))
    y_lim = _np.clip(_np.sort((y.min(),y.max())),_np.min(ext[1]),_np.max(ext[1]))
    #get DEM geotransform
    GT = MAP.GetGeoTransform()
    #Convert grid limits into DEM indices
    x_lim_idx = ((((x_lim - GT[0]) / GT[1]))) 
    y_lim_idx = ((((y_lim - GT[3]) / GT[5])))
    return x_lim, x_lim_idx, y_lim, y_lim_idx


def extract_extent(GD, ext):
    """
    This function extract a specified extent
    from a gdal dataset
    
    Parameters
    ----------
    GT : osgeo.gdal.Dataset
        The dataset of interest
    ext : iterable
        The extent specified in the form
        ((x_min, x_max), (y_min, y_max))
        
    Returns
    -------
        osgeo.gdal.Dataset
    """
    from osgeo import gdal
    #Get The Geotransform
    GT = GD.GetGeoTransform()
    #Define the new geotransform
    #the spacing is the same as the original,
    #the corner is specified by the new transform
    gt_new = [ext[0][0], GT[1], GT[2], ext[1][1], GT[4], GT[5]]
    #Now we compute the new size
    n_pix_x = int(_np.abs((ext[0][1] - ext[0][0]) / GT[1]))
    n_pix_y = int(_np.abs((ext[1][1] - ext[1][0]) / GT[5]))
    #Now, we can generate a new dataset
    mem_drv = gdal.GetDriverByName( 'MEM' )
    dest = mem_drv.Create('', n_pix_x, \
            n_pix_y, GD.RasterCount,\
            numeric_dt_to_gdal_dt(GD.GetRasterBand(1).DataType))
    dest.SetGeoTransform( gt_new )
    dest.SetProjection(GD.GetProjection())
    res = gdal.ReprojectImage( GD, dest, \
                GD.GetProjection(), GD.GetProjection(), \
                gdal.GRA_Bilinear)
    return dest


def gdal_to_np_format(arr):
    if arr.ndim == 2:
        return arr
    elif arr.ndim > 2:
        return arr.transpose([1,2,0])

def np_to_gdal_format(arr):
    if arr.ndim is 2:
        return arr
    else:
        return arr.transpose([1,2,0])


def segment_GT(DEM, center, S_l, heading):
    from osgeo import osr, gdal
    x_lim, x_lim_idx, y_lim, y_lim_idx = compute_map_extent(DEM, center, S_l, heading)
    #Now we cut the section of interest
    x_idx = _np.sort(x_lim_idx).astype(_np.int)
    y_idx = _np.sort(y_lim_idx).astype(_np.int)
    nx = _np.abs(_np.diff(x_idx))[0]
    ny = _np.abs(_np.diff(y_idx))[0]
    z = gdal_to_np_format(DEM.ReadAsArray(x_idx[0], y_idx[0] ,nx ,ny))
    #Now we save the DEM segment$
    GT = DEM.GetGeoTransform()
    GT_seg = list(GT)
    GT_seg[0] = x_lim[0]
    GT_seg[3] = y_lim[1]
    print(GT)
    proj = osr.SpatialReference()
    proj.ImportFromWkt(DEM.GetProjection())
    
    DEM_seg = write_gt(z, GT_seg, DEM.GetProjection())
    return DEM_seg


def gc_map_bi(DEM,center_TX,center_RX,S_l,heading, interp = None, seg_DEM = True):
    """
    This function computes a lookup-table
    that contains the radar coordinates
    corresponding to each coordinate in the
    DEM grid
    Parameters
    ----------
    DEM : osgeo.gdal.Dataset
        The DEM that will be used to compute the LUT
    S_l : pyrat.scatteringMatrix
        A scattering matrix object that contains
        the necessary data for geocoding (azimuth vector etc)
    center_TC : tuple
        the center of the radar in DEM coordinates
    center_RX : tuple
        the center of the radar in DEM coordinates
    heading : float
        The heading of the radar in degrees
    Returns
    -------
    lut : ndarray
        The real part of the lut contains the first index, the imaginary part the second
    incidence_angle : ndarray
        The computed incidence angle map
    
    """
    #%% DEM Segmentation
    #First of all, we compute the extent of the DEM
    #that is approx. covered by the radar
    x_lim, x_lim_idx, y_lim, y_lim_idx = compute_map_extent(DEM, center_RX, S_l, heading)
    if seg_DEM:
        DEM_seg = segment_GT(DEM, center_RX, S_l, heading)
    else:
        DEM_seg = DEM
    z = (DEM_seg.ReadAsArray()).astype(_np.float32)
    GT_seg = DEM_seg.GetGeoTransform()
    x = GT_seg[0] + _np.arange(0,DEM_seg.RasterXSize) * GT_seg[1]
    y = GT_seg[3] + _np.arange(0,DEM_seg.RasterYSize) * GT_seg[5]
    #Convert the positions to Radar Centered Coordinates
    #shift only
    x_rad_RX = x - center_RX[0]
    y_rad_RX = y - center_RX[1]
    z_rad_RX = z - center_RX[2] 
    x_rad_TX = x - center_TX[0]
    y_rad_TX = y - center_TX[1]
    z_rad_TX = z - center_TX[2]
    
    #The coordinates have to be rotated by the heading
    theta = heading
    #Compute rotation matrix to transform by heading
    R_mat = _np.array([[_np.cos(-theta), - _np.sin(-theta)],[_np.sin(-theta), _np.cos(-theta)]])
    xx, yy = _np.meshgrid(x_rad_RX,y_rad_RX)
    xy = _np.vstack([xx.flatten(),yy.flatten()]).transpose([1,0])
    xy_rot = _np.einsum('...ij,...j',R_mat,xy)
    x_rad_RX = xy_rot[:,0].reshape(z.shape)
    y_rad_RX = xy_rot[:,1].reshape(z.shape)
    xx, yy = _np.meshgrid(x_rad_TX,y_rad_TX)
    xy = _np.vstack([xx.flatten(),yy.flatten()]).transpose([1,0])
    xy_rot = _np.einsum('...ij,...j',R_mat,xy)
    x_rad_TX = xy_rot[:,0].reshape(z.shape)
    y_rad_TX = xy_rot[:,1].reshape(z.shape)
    #Compute bistatic ranges
    sep = _np.abs(_np.array(center_RX) - _np.array(center_TX))**2
    r_bi = _np.sqrt(x_rad_RX**2 + y_rad_RX**2 + z_rad_RX**2) + _np.sqrt(x_rad_TX**2 + y_rad_TX**2 + z_rad_TX**2) - sep 
    #Conver coordinates into range and azimuths
#    r_sl = _np.sqrt(x_rad**2 + y_rad**2 + z_rad**2)
    az = _np.arctan2(x_rad_RX, y_rad_RX)
    #Convert coordinates into indices
    r_step = S_l.r_vec[1] - S_l.r_vec[0]
    az_step = S_l.az_vec[1] - S_l.az_vec[0]
    r_idx = (r_bi - S_l.r_vec[0]) / r_step
    az_idx = (az - S_l.az_vec[0]) / (az_step)
    lut = 1j * az_idx + r_idx
#    #The slant ranges can be used to compute shadow maps
#    #Now we can compute the reverse transformation (From DEM to Radar)
#    #To each index (azimuth and range) in the radar geometry, the LUT contains
#    #the corresponding DEM indices
#    rev_lut = _np.zeros(S_l.shape)
#    r_vec = S_l.r_vec
#    az_vec =  S_l.az_vec - heading
#    r1,az1  = _np.meshgrid(r_vec, az_vec)
#    #Compute the points on the map that correspond to the slant ranges and azimuth on the radar
#    xrad = r1 * _np.sin(az1) + center[0]
#    yrad = r1 * _np.cos(az1) + center[1]
#    #Convert in map indices by using the geotransform
#    x_idx = (xrad - GT_seg[0]) / GT_seg[1]
#    y_idx = (yrad - GT_seg[3]) / GT_seg[5]
#    rev_lut = x_idx + 1j * y_idx
    return DEM_seg, lut
    


def gc_map(DEM,center,S_l,heading, interp = None, seg_DEM = True):
    """
    This function computes a lookup-table
    that contains the radar coordinates
    corresponding to each coordinate in the
    DEM grid
    Parameters
    ----------
    DEM : osgeo.gdal.Dataset
        The DEM that will be used to compute the LUT
    S_l : pyrat.scatteringMatrix
        A scattering matrix object that contains
        the necessary data for geocoding (azimuth vector etc)
    center : tuple
        the center of the radar in DEM coordinates
    heading : float
        The heading of the radar in degrees
    Returns
    -------
    lut : ndarray
        The real part of the lut contains the first index, the imaginary part the second
    incidence_angle : ndarray
        The computed incidence angle map
    
    """
    from osgeo import osr
    #%% DEM Segmentation
    #First of all, we compute the extent of the DEM
    #that is approx. covered by the radar
    x_lim, x_lim_idx, y_lim, y_lim_idx = compute_map_extent(DEM, center, S_l, heading)
    if seg_DEM:
        DEM_seg = segment_GT(DEM, center, S_l, heading)
    else:
        DEM_seg = DEM
    z = (DEM_seg.ReadAsArray()).astype(_np.float32)
    GT_seg = DEM_seg.GetGeoTransform()
    x = GT_seg[0] + _np.arange(0,DEM_seg.RasterXSize) * GT_seg[1]
    y = GT_seg[3] + _np.arange(0,DEM_seg.RasterYSize) * GT_seg[5]

    #Convert the positions to Radar Centered Coordinates
    #shift only
    x_rad = x - center[0]
    y_rad = y - center[1]
    z_rad = z - center[2] 
    #The coordinates have to be rotated by the heading
    theta = heading
    #Compute rotation matrix to transform by heading
    R_mat = _np.array([[_np.cos(-theta), - _np.sin(-theta)],[_np.sin(-theta), _np.cos(-theta)]])
    xx, yy = _np.meshgrid(x_rad,y_rad)
    xy = _np.vstack([xx.flatten(),yy.flatten()]).transpose([1,0])
    xy_rot = _np.einsum('...ij,...j',R_mat,xy)
    x_rad = xy_rot[:,0].reshape(z.shape)
    y_rad = xy_rot[:,1].reshape(z.shape)
    #Conver coordinates into range and azimuths
    r_sl = _np.sqrt(x_rad**2 + y_rad**2 + z_rad**2)
    az = _np.arctan2(x_rad, y_rad)
#    ia = _np.arcsin(z_rad / r_sl)
    #Convert coordinates into indices
    r_step = S_l.r_vec[1] - S_l.r_vec[0]
    az_step = S_l.az_vec[1] - S_l.az_vec[0]
    r_idx = (r_sl - S_l.r_vec[0]) / r_step
    az_idx = (az - S_l.az_vec[0]) / (az_step)
    lut = 1j * az_idx + r_idx
    #The slant ranges can be used to compute shadow maps
    #Now we can compute the reverse transformation (From DEM to Radar)
    #To each index (azimuth and range) in the radar geometry, the LUT contains
    #the corresponding DEM indices
    rev_lut = _np.zeros(S_l.shape)
    r_vec = S_l.r_vec
    az_vec =  S_l.az_vec - theta
    r1,az1  = _np.meshgrid(r_vec, az_vec)
    #Compute the points on the map that correspond to the slant ranges and azimuth on the radar
    xrad = r1 * _np.sin(az1) + center[0]
    yrad = r1 * _np.cos(az1) + center[1]
    #Convert in map indices by using the geotransform
    x_idx = (xrad - GT_seg[0]) / GT_seg[1]
    y_idx = (yrad - GT_seg[3]) / GT_seg[5]
    rev_lut = x_idx + 1j * y_idx
    #Compute the beta nought area given as the area of the
    #illuminated annulus
    r_d = r_vec[1] - r_vec[0]
    area_beta = (az_vec[1] - az_vec[0]) * ((r_vec + r_d)**2 - r_vec**2 )
    #Using the coordinates and the DEM in the radar centric cartesian system, we can compute the area etc
    zrad = bilinear_interpolate(z_rad,x_idx,y_idx)
    positions = _np.dstack((xrad, yrad, zrad))
    a = _np.roll(positions,1,axis = 0) - positions
    b = _np.roll(positions,1,axis = 1) - positions
    c = positions - _np.roll(positions,1,axis = 0)
    d = positions - _np.roll(positions,1,axis =1)
    #Compute and normalize normals
    c1 = _np.cross(a,b)
    c2 = _np.cross(c,d)
    normal = c1 / 2 + c2 / 2
    area = _np.linalg.norm(c1,axis = 2) /2 + _np.linalg.norm(c2,axis = 2)/2
    normal = normal / _np.linalg.norm(normal, axis = 2)[:,:,None]
    #Compute zenith angle
    u = _np.arccos(normal[:,:,2])
    #Compute incidence angle
    los_v = positions / _np.linalg.norm(positions, axis = 2)[:,:,None]
    dot = los_v[:,:,0] * normal[:,:,0] + los_v[:,:,1] * normal[:,:,1] + los_v[:,:,2] * normal[:,:,2]
    ia = _np.arccos(dot)
    #Compute the shadow map
    current = ia[:,0]
    shadow = _np.zeros_like(ia)
    for idx_r in range(ia.shape[1] - 1):
        nex = ia[:, idx_r + 1]  
        sh = nex < current
        shadow[:, idx_r] = sh
        current = nex * (sh) + current * ~sh
    return DEM_seg, lut, rev_lut, xrad, yrad, ia, area, area_beta, r_sl
    
def ly_sh_map(r_sl, ia):
    """
    This function computes the layover and shadow
    map from a DEM in the radar coordinatesd
    """
    r_current = r_sl[:,0] * 1
    ia_current = ia[:,0] * 1
    ly = _np.zeros_like(r_sl)
    sh = _np.zeros_like(r_sl)
    for idx_r in range(r_sl.shape[1] -1):
        r_next = r_sl[:, idx_r + 1]
        ia_next = ia[:, idx_r + 1]
        ia_current = ia[:, idx_r]
        ly_vec = r_current > r_next
        sh_vec = ia_next < ia_current
        r_current[ly_vec] = r_next[ly_vec]
        #if we dont have shadow, the next current incidence angle
        #equals the current next
        ia_current = ia_next * (~sh_vec) + ia_current * (sh_vec)
        r_current = r_next * (ly_vec) + r_current * (~ly_vec)
        ly[:, idx_r] = ly_vec
        sh[:, idx_r] = sh_vec
#        print ia_current
#        print ia_next
#        raw_input("Press enter to continue")
    return ly, sh
    
        
    
def reverse_lookup(image,lut):
    idx_az = _np.arange(image.shape[0])
    idx_r = _np.arange(image.shape[1])
    idx_az, idx_r = _np.meshgrid(idx_az, idx_r)
    

def coordinate_to_raster_index(gs, coordinate):
    GT = gs.GetGeoTransform()
    idx_x = _np.ceil((coordinate[0] - GT[0])/ GT[1])
    idx_y = _np.ceil((coordinate[1] - GT[3])/ GT[5])
    return (idx_x, idx_y)


def raster_index_to_coordinate(gs, index):
    GT = gs.GetGeoTransform()
    coord_x = index[0] * GT[1]  + GT[0]
    coord_y = index[1] * GT[5] + GT[3]
    return (coord_x, coord_y)
    
    
def bilinear_interpolate(im, x, y):
    
        
    x = _np.asarray(x)
    y = _np.asarray(y)

    x0 = _np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = _np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = _np.clip(x0, 0, im.shape[1]-1)
    x1 = _np.clip(x1, 0, im.shape[1]-1)
    y0 = _np.clip(y0, 0, im.shape[0]-1)
    y1 = _np.clip(y1, 0, im.shape[0]-1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    
    if im.ndim > 2:
        trailing_dim = im.ndim - 2
        access_vector = (Ellipsis,Ellipsis) + trailing_dim * (None,)
        wa = wa[access_vector]
        wb = wb[access_vector]
        wc = wc[access_vector]
        wd = wd[access_vector]
    interp = wa*Ia + wb*Ib + wc*Ic + wd*Id
    interp[y.astype(_np.long) >= im.shape[0] -1] = _np.nan
    interp[x.astype(_np.long) >= im.shape[1] - 1] = _np.nan
    interp[y.astype(_np.long) <= 0] = _np.nan
    interp[x.astype(_np.long) <= 0] = _np.nan
    return interp
    

def get_extent(ds):
    gt = ds.GetGeoTransform()
    gt_x_vec = tuple(_np.sort((gt[0], gt[0] + gt[1] * (ds.RasterXSize ))))
    gt_y_vec = tuple(_np.sort(((gt[3], gt[3] + gt[5] * (ds.RasterYSize)))))
    return gt_x_vec,gt_y_vec
    
    


def read_coordinate_extent(ds,coords, interp = None):
    """
    This function reads an extent from
    a raster using a tuple of map coordinates
    """
    gt  = ds.GetGeoTransform()
    px = ((coords[0] - gt[0]) / gt[1])
    py = ((coords[1] - gt[3]) / gt[5])
    RAS = ds.ReadAsArray()
    if RAS.ndim is 2:
        RAS = RAS
    else:
        RAS = RAS.transpose([1,2,0])
    if interp is None:
        if type(px) is float :
            px = _np.int(px)
            py = _np.int(py)
        else:
            px = _np.clip((px),0,RAS.shape[1] -1).astype(_np.int)
            py = _np.clip((py),0,RAS.shape[0] -1).astype(_np.int)
        return RAS[py,px]
    else:
        return interp(RAS,px,py)
    
def auto_heading(S, pixel_coord,geo_coord):
    geo_heading = _np.rad2deg(_np.arctan2(geo_coord[0], geo_coord[1]))
    pixel_az = S.az_vec[pixel_coord[0]]
    return pixel_az - geo_heading

def WGS84_to_LV95(center):
    from osgeo import osr
    WG = osr.SpatialReference(osr.GetWellKnownGeogCSAsWKT('WGS84'))
    CH = osr.SpatialReference()
    CH.ImportFromEPSG(2056)
    Tx = osr.CoordinateTransformation(WG,CH)
    center_1 = Tx.TransformPoint(center[1],center[0],center[2])
    return center_1
    

def geocode_image(image,pixel_size,*args):
    """
    This function converst a GPRI image in polar coordinates into cartesian coordinates.
    
    Parameters
    ----------
    image : ndarray
        The image to be converted.
    pixel_size : double
        The size of the pixel in the resulting cartesian image.
    args*
        List of arguments. Takes the first argument as the list of azimuth positions
        and the second as the corresponding list of range positions.
    Returns
    -------
    
    """

    try:
        r_vec = image.r_vec
        az_vec = image.az_vec
    except AttributeError:
        if len(args) > 0:
            r_vec = args[0]
            az_vec = args[1]
        else:
            raise TypeError
    #Image grid geometry
    r_max = _np.max(r_vec)
    r_min = _np.min(r_vec)
    az_max = _np.max(az_vec)
    az_min = _np.min(az_vec)
    az_step = _np.abs(az_vec[1] - az_vec[0])
    r_step = _np.abs(r_vec[1] - r_vec[0])
    #Compute desired grid
    az_vec_1 = _np.linspace(az_min,az_max,10)
    r_vec_1 = _np.linspace(r_min,r_max,10)
    bound_grid = _np.meshgrid(az_vec_1,r_vec_1)
    x = bound_grid[1] * _np.cos(bound_grid[0])
    y = bound_grid[1] * _np.sin(bound_grid[0])
    y_vec = (y.min(),y.max())
    x_vec = (x.min(),x.max())
    y_vec = _np.arange(y_vec[0],y_vec[1],pixel_size)
    x_vec = _np.arange(x_vec[0],x_vec[1],pixel_size)
    desired_grid = _np.meshgrid(x_vec,y_vec,indexing ='xy')
    desired_r = _np.sqrt(desired_grid[0]**2 + desired_grid[1]**2)
    desired_az = _np.arctan2(desired_grid[1],desired_grid[0])
    #Convert desired grid to indices
    az_idx = ((desired_az - az_min) / _np.double(az_step))
    r_idx = ((desired_r - r_min) / _np.double(r_step))
    r_idx = _np.clip(r_idx,0,image.shape[1]-1)
    az_idx = _np.clip(az_idx,0,image.shape[0]-1)
    az_idx = az_idx.astype(_np.float)
    r_idx = r_idx.astype(_np.float)
    gc = bilinear_interpolate(image,r_idx,az_idx)
    gc[az_idx.astype(_np.long) == image.shape[0] -1] = _np.nan
    gc[r_idx.astype(_np.long) == image.shape[1] - 1] = _np.nan
    gc[az_idx.astype(_np.long) == 0] = _np.nan
    gc[r_idx.astype(_np.long) == 0] = _np.nan
    return gc, x_vec, y_vec
    
def pauli_rgb(scattering_vector, normalized= False, pl=True, gamma = 1, sf = 2.5):
        """
        This function produces a rgb image from a scattering vector.
        
        Parameters
        ----------
        scattering_vector : ndarray 
            the scattering vector to be represented.
        normalized : bool
            set to true for the relative rgb image, where each channel is normalized by the sum.
        log : bool
            set to True to display the channels in logarithmic form.
        """
        if not normalized:
            if pl:
                data_diagonal = _np.abs(scattering_vector)
                sp = _np.sum(data_diagonal,axis = 2)
#                m = _np.nanmean(sp, axis = (0,1)) * sf
#                m = _np.nanmax(sp, axis = (0,1)) * sf
                sp[_np.isnan(sp)] = 0
                min_perc = []
                max_perc = []
                for idx in range(3):
                    min_perc.append(_np.percentile(sp / 3, 5))
                    max_perc.append(_np.percentile(sp / 3, 99.5))
#                mx = _np.nanmax(data_diagonal, axis = (0,1))
#                m = _np.minimum(m,mx) 
                data_diagonal[:,:,0] = scale_array(data_diagonal[:,:,0], 
                min_val = min_perc[0], max_val = max_perc[0])
                data_diagonal[:,:,1] = scale_array(data_diagonal[:,:,1], 
                min_val = min_perc[1], max_val = max_perc[1])
                data_diagonal[:,:,2] = scale_array(data_diagonal[:,:,2], 
                min_val = min_perc[2], max_val = max_perc[2])
                data_diagonal = (((data_diagonal)**(gamma)))
                data_diagonal[:,:,0] = scale_array(data_diagonal[:,:,0])
                data_diagonal[:,:,1] = scale_array(data_diagonal[:,:,1])
                data_diagonal[:,:,2] = scale_array(data_diagonal[:,:,2])
            else:
                R = data_diagonal[:,:,0] / _np.nanmax(data_diagonal[:,:,0])
                G = data_diagonal[:,:,1] / _np.nanmax(data_diagonal[:,:,1])
                B = data_diagonal[:,:,2] / _np.nanmax(data_diagonal[:,:,2])
                out = 1 - _np.exp(- data_diagonal * _np.array(k)[None,None, :])
                return out
            R = data_diagonal[:,:,0]
            G = data_diagonal[:,:,1]
            B = data_diagonal[:,:,2]
            out = _np.dstack((R,G,B))
        else:
            span = _np.sum(scattering_vector,axis=2)
            out = _np.abs(scattering_vector  /span[:,:,None])
        return out
        
def show_geocoded(geocoded_image,**kwargs):
        """
        This function is a wrapper to call imshow with a 
        list produced by the geocode_image function.
        ----------
        geocoded_image_list : list 
            list containing the geocoded image and the x and y vectors of the new grid.
        """
        ax = _plt.gca()
        a = geocoded_image
        if a.ndim is 3 and  a.shape[-1] == 3:
            alpha = _np.isnan(a[:,:,0])
            a = _np.dstack((a,~alpha))
        else:
            a = _np.ma.masked_where(_np.isnan(a), a)
        _plt.imshow(a ,**kwargs)
        _plt.xlabel(r'Easting [m]')
        _plt.ylabel(r'Northing [m]')
        return a
class ROI:
    """
    Class to represent ROIS
    """
    def __init__(*args):
        self = args[0]
        if type(args[1]) is list:
            self.polygon_list = args[1]
        else:
            self.polygon_list = [args[1]]
        self.shape = args[2]
        self.mask = _np.zeros(self.shape,dtype=_np.bool)
        self.fill_mask()

        
    def fill_mask(self):
        def dstack_product(x, y):
            return _np.dstack(_np.meshgrid(x, y)).reshape(-1, 2)
        x_grid = _np.arange(self.shape[0])
        y_grid = _np.arange(self.shape[1])
        points = dstack_product(y_grid, x_grid)
        for pt in self.polygon_list:
            path = _mpl.path.Path(pt)
            self.mask = self.mask + path.contains_points(points, radius = 0.5).reshape(self.shape)
            
    def draw_mask(self,ax,**kwargs):
        display_to_ax = ax.transAxes.inverted().transform
        data_to_display = ax.transData.transform
        for dta_pts in self.polygon_list:
            ax_pts = display_to_ax(data_to_display(dta_pts))
            p = _plt.Polygon(ax_pts, True, transform=ax.transAxes,**kwargs)
            ax.add_patch(p)


def ROC(cases,scores,n_positive, n_negative):
    sort_indices = _np.argsort(scores)
    sort_cases = cases[sort_indices[::-1]]
    pd = _np.cumsum(sort_cases == 1) / _np.double(n_positive)
    pf = _np.cumsum(sort_cases == 0) / _np.double(n_negative)
    return pf, pd


def confusionMatrix(image,masks,training_areas,function, threshold):
    #Compute classes taking the mean of the class pixels
    classes = ()
    for mask_idx, mask in enumerate(training_areas):
        classes = classes + (_np.mean(image[mask],axis = 0),)
    classified, distance = pyrat.classifier(image,classes, function, threshold)
    cm_1 = _np.zeros((len(masks),len(masks)))
    for idx_1,mask in enumerate(masks):
        for idx_2,mask1 in enumerate(masks):
            cm_1[idx_1,idx_2] = _np.sum(classified[mask1] == idx_1 + 1) / _np.double(_np.sum(mask1 == True))
    return cm_1, classified, distance
    
def rectangle_vertices(v1,v2):
    x1 = v1[1]
    y1 = v1[0]
    x2 = v2[1]
    y2 = v2[0]
    return _np.array([[x1,y1],[x1,y2],[x2,y2],[x2,y1]])

def scale_coherence(c):
#    c_sc = _np.select(((_np.sin(c * _np.pi / 2)), 0.3), (c > 0.2, c<0.2))
    return _np.sin(c * _np.pi / 2)


  
def disp_mph(data, dt = 'amplitude', k = 0.5, min_val = -_np.pi ,
             max_val =  _np.pi, return_pal = False):
    H = scale_array(_np.angle(data), min_val = min_val, max_val = max_val)
    S = _np.zeros_like(H) + 0.5
    if dt == 'coherence':
        V = scale_coherence((_np.abs(data)))
    elif dt == 'amplitude':
        V = scale_array(exp_im(data,k))
    elif dt == 'none':
        V = scale_array(_np.abs(data))
    RGB = _mpl.colors.hsv_to_rgb(_np.dstack((H, S, V)))
    if return_pal:
        H_pal = scale_array(_np.linspace(min_val, max_val, 255))
        S_pal = H_pal * 0 + 0.8
        V_pal = H_pal * 0 + 1 
        pal = _mpl.colors.hsv_to_rgb(_np.dstack((H_pal, S_pal, V_pal))).squeeze()
        cmap = _mpl.colors.LinearSegmentedColormap.from_list('my_colormap',pal,256)
        return RGB, cmap
    else:
        return RGB

def load_custom_palette():
    RGB = disp_mph(_np.exp(1j * _np.linspace(0,2 * _np.pi,255))).squeeze()
    cmap = _mpl.colors.LinearSegmentedColormap.from_list('my_colormap',RGB,256)
    return cmap
     
def extract_section(image, center, size):
    x = center[0] + _np.arange(-int(size[0] / 2.0),int(size[0] / 2.0))
    y = center[1] + _np.arange(-int(size[0] / 2.0),int(size[0] / 2.0))
    x = _np.mod(x, image.shape[0])  
    y = _np.mod(y, image.shape[1])
    xx, yy = _np.meshgrid(x, y)
    return image[xx,yy]
    
def show_if(S1, S2, win):
    name_list = ['HH', 'HV', 'VH', 'VV']
    k1 = S1.scattering_vector(basis='lexicographic')
    k2 = S2.scattering_vector(basis='lexicographic')
    if_mat = _np.zeros(S1.shape[0:2] + (4,4), dtype = _np.complex64)
    for i in range(4):
        for j in range(4):
            c_if = pyrat.coherence(k1[:, :, i], k2[:, :, j], win)
            if_mat[:,:,i,j] = c_if
            RGB = if_hsv(c_if)
            if i == 0 and j == 0:
                ax =  _plt.subplot2grid((4, 4), (i, j))
                _plt.imshow(RGB,  cmap = 'gist_rainbow', interpolation = 'none')
                ax = _plt.gca()
            else:
                 _plt.subplot2grid((4, 4), (i, j))
                 _plt.imshow(RGB , cmap = 'gist_rainbow', interpolation = 'none')
            _plt.title(name_list[i] + name_list[j])
    return if_mat
            
def hsv_cp(H,alpha,span):
    V = scale_array(_np.log10(span))    
    H1 = scale_array(alpha, top = 0, bottom = 240)  / 360
    S = 1 - H
    return _mpl.colors.hsv_to_rgb(_np.dstack((H1,S,V)))

def show_signature(signature_output, rotate = False):
    phi = signature_output[2]
    tau = signature_output[3]
    sig_co = signature_output[0]
    sig_cross = signature_output[1]
    phi_1, tau_1 = _np.meshgrid(phi, tau)
    if rotate:
        xt = [-90,90,-45,45]
        sig_co = sig_co.T
        sig_cross = sig_cross.T
    else:
        xt = [-45,45,-90,90]
    f_co = _plt.figure()
    _plt.imshow(sig_co, interpolation = 'none', cmap =  'RdBu_r', extent = xt)
    _plt.locator_params(nbins=5)
    _plt.xlabel(r'ellipicity $\tau$')
    _plt.ylabel(r'orientation $\phi$')
    _plt.xticks(rotation=90)

#    _plt.axis('equal')
    f_x = _plt.figure()
    _plt.imshow(sig_cross, interpolation = 'none', cmap =  'RdBu_r', extent = xt)
    _plt.locator_params(nbins=5)
    _plt.xlabel(r'ellipicity $\tau$')
    _plt.ylabel(r'orientation $\phi$')
    _plt.xticks(rotation=90)

#    _plt.axis('equal')
    return f_co, f_x


def blockshaped(arr, n_patch):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    from:

    """
    nrows, ncols = n_patch
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

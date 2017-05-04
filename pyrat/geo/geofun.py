import itertools as _iter

import numpy as _np
import scipy.ndimage as _ndim
from osgeo import osr as _osr, gdal, ogr as _ogr

from . import transforms as _transf
from pyrat.visualization.visfun import bilinear_interpolate
from ..fileutils import gpri_files as _gpf
from ..fileutils import parameters as _params


import matplotlib.pyplot as plt
import collections as _coll

import gdal
import gdalnumeric

def copy_and_modify_gt(RAS, gt):
    from osgeo import gdal

    mem_drv = gdal.GetDriverByName('MEM')
    output_dataset = mem_drv.Create('', RAS.shape[1], RAS.shape[0], RAS.shape[2], gdal.GDT_Float32)
    output_dataset.SetProjection(gt.GetProjection())
    output_dataset.SetGeoTransform(gt.GetGeoTransform())
    for n_band in range(RAS.shape[-1] - 1):
        output_dataset.GetRasterBand(n_band + 1).WriteArray(RAS[:, :, n_band])
    return output_dataset

def save_raster(ds, path):
    """
    Writes a :obj:`osgeo.ogr.DataSource` as a geotiff file
    Parameters
    ----------
    ds
    path

    Returns
    -------

    """
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        path,
        ds.RasterXSize,
        ds.RasterYSize,
        ds.RasterCount,
        ds.GetRasterBand(1).DataType)
    print(dataset)
    dataset.SetGeoTransform(ds.GetGeoTransform())
    dataset.SetProjection(ds.GetProjection())
    for band in range(ds.RasterCount):
        dataset.GetRasterBand(band + 1).WriteArray(ds.GetRasterBand(band+1).ReadAsArray())
        dataset.FlushCache()  # Write to disk.

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
    # Remove nans
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
        for n_band in range(nchan - 1):
            band = raster[:, :, n_band]
            dataset.GetRasterBand(n_band + 1).WriteArray(band)
            dataset.FlushCache()  # Write to disk.
    else:
        dataset.GetRasterBand(1).WriteArray(raster)
        dataset.FlushCache()  # Write to disk.
    # dataset.GDALClose()
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
    mem_drv = gdal.GetDriverByName('MEM')
    ds_copy = mem_drv.CreateCopy('', ds)
    ds_copy.SetGeoTransform(GT_new)
    return ds_copy


def numeric_dt_to_gdal_dt(number):
    from osgeo import gdal
    print(number)
    conversion_dict = {
        1: gdal.GDT_Byte,
        2: gdal.GDT_UInt16,
        3: gdal.GDT_Int16,
        4: gdal.GDT_UInt32,
        5: gdal.GDT_Int32,
        6: gdal.GDT_Float32,
        7: gdal.GDT_Float64,
        10: gdal.GDT_CFloat64,
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
        DOM_sec = segment_dem_radar(DOM, center, sec, heading)
        DOM_sec = resample_DEM(DOM_sec, [0.25, -0.25])
        DEM_seg, lut_sec, rev_lut_sec, xrad, yrad, ia, area, area_beta, r_sl = gc_map(DOM_sec, center,
                                                                                      sec, heading, seg_DEM=False)
        # Find maximum amplitude
        ampl = _np.abs(sec[:, :, 0, 0])
        max_idx = _np.nanargmax(ampl)
        max_pos = _np.unravel_index(max_idx, sec.shape[0:2])
        # Convert position into coordinate
        max_pos_gc = rev_lut_sec[max_pos]
        max_pos_gc = (max_pos_gc.real, max_pos_gc.imag)
        max_coord = raster_index_to_coordinate(DOM_sec, max_pos_gc)
        err = _np.array(tiepoint[0:2]) - _np.array(max_coord)
        return _np.linalg.norm(err)

    import scipy

    res = scipy.optimize.minimize_scalar(ftm, bounds=(0, 2 * _np.pi))
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

    tx = osr.CoordinateTransformation(osr.SpatialReference(gt_to_project.GetProjection()),
                                      osr.SpatialReference(gt_reference.GetProjection()))
    geo_t = gt_to_project.GetGeoTransform()
    geo_t_ref = gt_reference.GetGeoTransform()
    x_size = gt_reference.RasterXSize  # Raster xsize
    y_size = gt_reference.RasterYSize  # Raster ysize
    (ulx, uly, ulz) = tx.TransformPoint(geo_t[0], geo_t[3])
    (lrx, lry, lrz) = tx.TransformPoint(geo_t[0] + geo_t_ref[1] * x_size,
                                        geo_t[3] + geo_t_ref[5] * y_size)
    # Compute new geotransform
    new_geo = (geo_t_ref[0], geo_t_ref[1], geo_t[2],
               geo_t_ref[3], geo_t_ref[4], geo_t_ref[5])
    mem_drv = gdal.GetDriverByName('MEM')
    pixel_spacing_x = geo_t_ref[1]
    pixel_spacing_y = geo_t_ref[5]
    dest = mem_drv.Create('', int((lrx - ulx) / _np.abs(pixel_spacing_x)),
                          int((uly - lry) / _np.abs(pixel_spacing_y)), gt_to_project.RasterCount,
                          gt_to_project.GetRasterBand(1).DataType)
    dest.SetGeoTransform(new_geo)
    dest.SetProjection(gt_reference.GetProjection())
    res = gdal.ReprojectImage(gt_to_project, dest,
                              gt_to_project.GetProjection(), gt_reference.GetProjection(),
                              gdal.GRA_Bilinear)
    return dest


def reproject_radar(S, S_ref):
    """
    This function reprojects
    a radar image
    to the sample range and azimuth spacing
    of a given image and coregisters them
    """
    # Azimuth and range spacings
    az_sp = S.az_vec[1] - S.az_vec[0]
    az_sp_ref = S_ref.az_vec[1] - S_ref.az_vec[0]
    rg_sp = S.r_vec[1] - S.r_vec[0]
    rg_sp_ref = S_ref.r_vec[1] - S_ref.r_vec[0]
    # Compute new azimuth and range vectors
    az_vec_new = S_ref.az_vec
    rg_vec_new = S_ref.r_vec
    az_idx = _np.arange(S_ref.shape[0]) * az_sp_ref / az_sp
    rg_idx = _np.arange(S_ref.shape[1]) * rg_sp_ref / rg_sp
    az, r = _np.meshgrid(az_idx, rg_idx, order='ij')
    int_arr = bilinear_interpolate(S, r.T, az.T).astype(_np.complex64)
    S_res = S.__array_wrap__(int_arr)
    S_res.az_vec = az_vec_new
    S_res.r_vec = rg_vec_new
    return S_res


def WGS84_to_LV95(center):
    from osgeo import osr
    WG = osr.SpatialReference(osr.GetWellKnownGeogCSAsWKT('WGS84'))
    CH = osr.SpatialReference()
    CH.ImportFromEPSG(21781)
    Tx = osr.CoordinateTransformation(WG, CH)
    center_1 = Tx.TransformPoint(center[1], center[0], center[2])
    return center_1


def compute_radar_data_extent(MAP, center, S, heading, return_coverage=False):
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
    # Compute the area covered by radar
    # in DEM coordinates
    r_vec = (S.r_vec.max(), S.r_vec.min())
    az_vec = _np.deg2rad(S.az_vec + heading)
    r, az = _np.meshgrid(r_vec, az_vec)
    x = r * _np.sin(az) + center[0]
    y = r * _np.cos(az) + center[1]
    ext = (x.min(),x.max(), y.min(), y.max())
    return ext
    # # Compute limits of area covered and clip to the DEM size
    # x_lim = _np.clip(_np.sort((x.min(), x.max())), _np.min(ext[0:2]), _np.max(ext[0:2]))
    # y_lim = _np.clip(_np.sort((y.min(), y.max())), _np.min(ext[2:4]), _np.max(ext[2:4]))
    # # get DEM geotransform
    # GT = MAP.GetGeoTransform()
    # # Convert grid limits into DEM indices
    # x_lim_idx = ((x_lim - GT[0]) / GT[1])
    # y_lim_idx = ((y_lim - GT[3]) / GT[5])
    # if return_coverage:
    #     return x_lim, x_lim_idx, y_lim, y_lim_idx, x, y
    # else:
    #     return x_lim, x_lim_idx, y_lim, y_lim_idx


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
        (x_min, x_max,y_min, y_max)

    Returns
    -------
        osgeo.gdal.Dataset
    """
    from osgeo import gdal
    # Get The Geotransform
    GT = GD.GetGeoTransform()
    # Define the new geotransform
    # the spacing is the same as the original,
    # the corner is specified by the new transform
    gt_new = [ext[0], GT[1], GT[2], ext[2], GT[4], GT[5]]
    # Now we compute the new size
    n_pix_x = int(_np.abs((ext[1] - ext[0]) / GT[1]))
    n_pix_y = int(_np.abs((ext[3] - ext[2]) / GT[5]))
    # Now, we can generate a new dataset
    mem_drv = gdal.GetDriverByName('MEM')
    dest = mem_drv.Create('', n_pix_x,
                          n_pix_y, GD.RasterCount,
                          numeric_dt_to_gdal_dt(GD.GetRasterBand(1).DataType))
    dest.SetGeoTransform(gt_new)
    dest.SetProjection(GD.GetProjection())
    res = gdal.ReprojectImage(GD, dest,
                              GD.GetProjection(), GD.GetProjection(),
                              gdal.GRA_Bilinear)
    return dest


def gdal_to_np_format(arr):
    if arr.ndim == 2:
        return arr
    elif arr.ndim > 2:
        return arr.transpose([1, 2, 0])


def np_to_gdal_format(arr):
    if arr.ndim is 2:
        return arr
    else:
        return arr.transpose([1, 2, 0])


def segment_dem_radar(DEM, center, S_l, heading):
    ext = compute_radar_data_extent(DEM, center, S_l, heading)
    corners = extent_to_corners(ext)
    print(corners)
    sl = (slice(corners[0][0],corners[1][0]),slice(corners[0][1],corners[1][1]))
    #Additional dims
    additional_dims = DEM.RasterCount
    segmented = DEM.ReadAsArray()[(None,)*additional_dims + sl]
    print(segmented.shape)
    plt.imshow(segmented[0,:,:])
    plt.show()
    # # Now we cut the section of interest
    # x_idx = _np.sort(x_lim_idx).astype(_np.int)
    # y_idx = _np.sort(y_lim_idx).astype(_np.int)
    # nx = _np.abs(_np.diff(x_idx))[0]
    # ny = _np.abs(_np.diff(y_idx))[0]
    # print(x_idx[0],y_idx[0])
    # z =  gdal_to_np_format(DEM.ReadAsArray())
    # print(z)
    # # z = gdal_to_np_format(DEM.ReadAsArray(x_idx[0], y_idx[0], nx, ny))
    # # Now we save the DEM segment$
    # GT = DEM.GetGeoTransform()
    # GT_seg = list(GT)
    # GT_seg[0] = x_lim[0]
    # GT_seg[3] = y_lim[1]
    # proj = osr.SpatialReference()
    # proj.ImportFromWkt(DEM.GetProjection())
    # DEM_seg = write_gt(z, GT_seg, DEM.GetProjection())
    return segmented


def gc_map_bi(DEM, center_TX, center_RX, S_l, heading, interp=None, seg_DEM=True):
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
    # %% DEM Segmentation
    # First of all, we compute the extent of the DEM
    # that is approx. covered by the radar
    x_lim, x_lim_idx, y_lim, y_lim_idx = compute_radar_data_extent(DEM, center_RX, S_l, heading)
    if seg_DEM:
        DEM_seg = segment_dem_radar(DEM, center_RX, S_l, heading)
    else:
        DEM_seg = DEM
    z = (DEM_seg.ReadAsArray()).astype(_np.float32)
    GT_seg = DEM_seg.GetGeoTransform()
    x = GT_seg[0] + _np.arange(0, DEM_seg.RasterXSize) * GT_seg[1]
    y = GT_seg[3] + _np.arange(0, DEM_seg.RasterYSize) * GT_seg[5]
    # Convert the positions to Radar Centered Coordinates
    # shift only
    x_rad_RX = x - center_RX[0]
    y_rad_RX = y - center_RX[1]
    z_rad_RX = z - center_RX[2]
    x_rad_TX = x - center_TX[0]
    y_rad_TX = y - center_TX[1]
    z_rad_TX = z - center_TX[2]

    # The coordinates have to be rotated by the heading
    theta = heading
    # Compute rotation matrix to transform by heading
    R_mat = _np.array([[_np.cos(-theta), - _np.sin(-theta)], [_np.sin(-theta), _np.cos(-theta)]])
    xx, yy = _np.meshgrid(x_rad_RX, y_rad_RX)
    xy = _np.vstack([xx.flatten(), yy.flatten()]).transpose([1, 0])
    xy_rot = _np.einsum('...ij,...j', R_mat, xy)
    x_rad_RX = xy_rot[:, 0].reshape(z.shape)
    y_rad_RX = xy_rot[:, 1].reshape(z.shape)
    xx, yy = _np.meshgrid(x_rad_TX, y_rad_TX)
    xy = _np.vstack([xx.flatten(), yy.flatten()]).transpose([1, 0])
    xy_rot = _np.einsum('...ij,...j', R_mat, xy)
    x_rad_TX = xy_rot[:, 0].reshape(z.shape)
    y_rad_TX = xy_rot[:, 1].reshape(z.shape)
    # Compute bistatic ranges
    sep = _np.abs(_np.array(center_RX) - _np.array(center_TX)) ** 2
    r_bi = _np.sqrt(x_rad_RX ** 2 + y_rad_RX ** 2 + z_rad_RX ** 2) + _np.sqrt(
        x_rad_TX ** 2 + y_rad_TX ** 2 + z_rad_TX ** 2) - sep
    # Conver coordinates into range and azimuths
    #    r_sl = _np.sqrt(x_rad**2 + y_rad**2 + z_rad**2)
    az = _np.arctan2(x_rad_RX, y_rad_RX)
    # Convert coordinates into indices
    r_step = S_l.r_vec[1] - S_l.r_vec[0]
    az_step = S_l.az_vec[1] - S_l.az_vec[0]
    r_idx = (r_bi - S_l.r_vec[0]) / r_step
    az_idx = (az - S_l.az_vec[0]) / az_step
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


def gc_map(DEM, S_l, interp=None, seg_DEM=True, heading=0):
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
        The heading of the radar in radians
    Returns
    -------
    lut : ndarray
        The real part of the lut contains the first index, the imaginary part the second
    incidence_angle : ndarray
        The computed incidence angle map

    """
    # %% DEM Segmentation
    #Compute radar location
    center = WGS84_to_LV95([S_l.GPRI_ref_north, S_l.GPRI_ref_east, S_l.GPRI_ref_alt ])
    # First of all, we compute the extent of the DEM
    if seg_DEM:
        DEM_seg = segment_dem_radar(DEM, center, S_l, heading)
    else:
        DEM_seg = DEM
    z = (DEM_seg.ReadAsArray()).astype(_np.float32)
    GT_seg = DEM_seg.GetGeoTransform()
    x = GT_seg[0] + _np.arange(0, DEM_seg.RasterXSize) * GT_seg[1]
    y = GT_seg[3] + _np.arange(0, DEM_seg.RasterYSize) * GT_seg[5]

    # Convert the positions to Radar Centered Coordinates
    # shift only
    x_rad = x - center[0]
    y_rad = y - center[1]
    z_rad = z - center[2]
    # The coordinates have to be rotated by the heading
    theta = heading
    # Compute rotation matrix to transform by heading
    R_mat = _np.array([[_np.cos(-theta), - _np.sin(-theta)], [_np.sin(-theta), _np.cos(-theta)]])
    xx, yy = _np.meshgrid(x_rad, y_rad)
    xy = _np.vstack([xx.flatten(), yy.flatten()]).transpose([1, 0])
    xy_rot = _np.einsum('...ij,...j', R_mat, xy)
    x_rad = xy_rot[:, 0].reshape(z.shape)
    y_rad = xy_rot[:, 1].reshape(z.shape)
    # Conver coordinates into range and azimuths
    r_sl = _np.sqrt(x_rad ** 2 + y_rad ** 2 + z_rad ** 2)
    az = _np.arctan2(x_rad, y_rad)
    #    ia = _np.arcsin(z_rad / r_sl)
    # Convert coordinates into indices
    r_step = S_l.r_vec[1] - S_l.r_vec[0]
    az_step = S_l.az_vec[1] - S_l.az_vec[0]
    r_idx = (r_sl - S_l.r_vec[0]) / r_step
    az_idx = (az - S_l.az_vec[0]) / az_step
    lut = 1j * az_idx + r_idx
    # The slant ranges can be used to compute shadow maps
    # Now we can compute the reverse transformation (From DEM to Radar)
    # To each index (azimuth and range) in the radar geometry, the LUT contains
    # the corresponding DEM indices
    rev_lut = _np.zeros(S_l.shape)
    r_vec = S_l.r_vec
    az_vec = S_l.az_vec - theta
    r1, az1 = _np.meshgrid(r_vec, az_vec)
    # Compute the points on the map that correspond to the slant ranges and azimuth on the radar
    xrad = r1 * _np.sin(az1) + center[0]
    yrad = r1 * _np.cos(az1) + center[1]
    # Convert in map indices by using the geotransform
    x_idx = (xrad - GT_seg[0]) / GT_seg[1]
    y_idx = (yrad - GT_seg[3]) / GT_seg[5]
    rev_lut = x_idx + 1j * y_idx
    # # Compute the beta nought area given as the area of the
    # # illuminated annulus
    # r_d = r_vec[1] - r_vec[0]
    # area_beta = (az_vec[1] - az_vec[0]) * ((r_vec + r_d) ** 2 - r_vec ** 2)
    # # Using the coordinates and the DEM in the radar centric cartesian system, we can compute the area etc
    # zrad = bilinear_interpolate(z_rad, x_idx, y_idx)
    # positions = _np.dstack((xrad, yrad, zrad))
    # a = _np.roll(positions, 1, axis=0) - positions
    # b = _np.roll(positions, 1, axis=1) - positions
    # c = positions - _np.roll(positions, 1, axis=0)
    # d = positions - _np.roll(positions, 1, axis=1)
    # # Compute and normalize normals
    # c1 = _np.cross(a, b)
    # c2 = _np.cross(c, d)
    # normal = c1 / 2 + c2 / 2
    # area = _np.linalg.norm(c1, axis=2) / 2 + _np.linalg.norm(c2, axis=2) / 2
    # normal = normal / _np.linalg.norm(normal, axis=2)[:, :, None]
    # # Compute zenith angle
    # u = _np.arccos(normal[:, :, 2])
    # # Compute incidence angle
    # los_v = positions / _np.linalg.norm(positions, axis=2)[:, :, None]
    # dot = los_v[:, :, 0] * normal[:, :, 0] + los_v[:, :, 1] * normal[:, :, 1] + los_v[:, :, 2] * normal[:, :, 2]
    # #    ia = _np.arccos(dot)
    # ia = _np.arcsin(zrad / _np.linalg.norm(positions, axis=2))
    # # Compute the shadow map
    # current = ia[:, 0]
    # shadow = _np.zeros_like(ia)
    # for idx_r in range(ia.shape[1] - 1):
    #     nex = ia[:, idx_r + 1]
    #     sh = nex < current
    #     shadow[:, idx_r] = sh
    #     current = nex * sh + current * ~sh
    return DEM_seg, lut, rev_lut


def ly_sh_map(r_sl, ia):
    """
    This function computes the layover and shadow
    map from a DEM in the radar coordinatesd
    """
    r_current = r_sl[:, 0] * 1
    ia_current = ia[:, 0] * 1
    ly = _np.zeros_like(r_sl)
    sh = _np.zeros_like(r_sl)
    for idx_r in range(r_sl.shape[1] - 1):
        r_next = r_sl[:, idx_r + 1]
        ia_next = ia[:, idx_r + 1]
        ia_current = ia[:, idx_r]
        ly_vec = r_current > r_next
        sh_vec = ia_next < ia_current
        r_current[ly_vec] = r_next[ly_vec]
        # if we dont have shadow, the next current incidence angle
        # equals the current next
        ia_current = ia_next * (~sh_vec) + ia_current * sh_vec
        r_current = r_next * ly_vec + r_current * (~ly_vec)
        ly[:, idx_r] = ly_vec
        sh[:, idx_r] = sh_vec
    # print ia_current
    #        print ia_next
    #        raw_input("Press enter to continue")
    return ly, sh


def coordinate_to_raster_index(GT, coordinate):
    idx_x = _np.ceil((coordinate[0] - GT[0]) / GT[1])
    idx_y = _np.ceil((coordinate[1] - GT[3]) / GT[5])
    return idx_x, idx_y


def raster_index_to_coordinate(GT, index):
    coord_x = index[0] * GT[1] + GT[0]
    coord_y = index[1] * GT[5] + GT[3]
    return coord_x, coord_y


def get_dem_extent(par_dict):
    gt = [par_dict['corner_east'][0], par_dict['post_east'][0],
          0, 0, par_dict['corner_north'][0], par_dict['post_north'][0]]
    gt_x_vec = tuple(_np.sort((gt[0], gt[0] + gt[1] * (par_dict['width']))))
    gt_y_vec = tuple(_np.sort((gt[3], gt[3] + gt[5] * (par_dict['nlines']))))
    return gt_x_vec[0], gt_x_vec[1], gt_y_vec[0], gt_y_vec[1]


def gt_from_dem_par(par_dict):
    """"
        This function computes the geotransform parameters from a gamma DEM parameters file

    """
    gt = [par_dict['corner_east'], par_dict['post_east'],
          0, 0, par_dict['corner_north'], par_dict['post_north']]
    return gt


def read_coordinate_extent(ds, coords, interp=None):
    """
    This function reads an extent from
    a raster using a tuple of map coordinates
    """
    gt = ds.GetGeoTransform()
    px = ((coords[0] - gt[0]) / gt[1])
    py = ((coords[1] - gt[3]) / gt[5])
    RAS = ds.ReadAsArray()
    if RAS.ndim is 2:
        RAS = RAS
    else:
        RAS = RAS.transpose([1, 2, 0])
    if interp is None:
        if type(px) is float:
            px = _np.int(px)
            py = _np.int(py)
        else:
            px = _np.clip(px, 0, RAS.shape[1] - 1).astype(_np.int)
            py = _np.clip(py, 0, RAS.shape[0] - 1).astype(_np.int)
        return RAS[py, px]
    else:
        return interp(RAS, px, py)


def direct_lut(image, pixel_size):
    r_vec = image.r_vec
    az_vec = _np.deg2rad(image.az_vec)
    # Compute inverse LUT (radar coordinate -> image coordinates)
    rr, zz = _np.meshgrid(r_vec, az_vec, indexing='xy')
    xx = (rr * _np.cos(zz))
    yy = (rr * _np.sin(zz))
    # xx and yy contain Coordinates in image space (in meters!) that correspond to each radar pixel
    # we need to convert it into pixels
    xx = (xx - xx.min()) / pixel_size
    yy = (yy - yy.min()) / pixel_size
    return xx + 1j * yy


def geocode_image(image, pixel_size, *args):
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
        az_vec = _np.deg2rad(image.az_vec)
    except AttributeError:
        if len(args) > 0:
            r_vec = args[0]
            az_vec = _np.deg2rad(args[1])
        else:
            raise TypeError
    # Desired grid
    # Image grid geometry
    r_max = _np.max(r_vec)
    r_min = _np.min(r_vec)
    az_max = _np.max(az_vec)
    az_min = _np.min(az_vec)
    az_step = _np.abs(az_vec[1] - az_vec[0])
    az_sign = -_np.sign(az_vec[1] - az_vec[0])
    r_step = _np.abs(r_vec[1] - r_vec[0])
    # Compute desired grid
    az_vec_1 = _np.linspace(az_min, az_max, num=8)
    r_vec_1 = _np.linspace(r_min, r_max, num=8)
    bound_grid = _np.meshgrid(az_vec_1, r_vec_1)
    x = bound_grid[1] * _np.cos(bound_grid[0])
    y = bound_grid[1] * _np.sin(bound_grid[0])
    # Determine bounds
    y_vec = (y.min(), y.max())
    x_vec = (x.min(), x.max())
    y_vec = _np.arange(y_vec[0], y_vec[1], pixel_size)
    x_vec = _np.arange(x_vec[0], x_vec[1], pixel_size)
    # Grid of desired pixels
    desired_grid = _np.meshgrid(x_vec, y_vec, indexing='xy')
    desired_r = _np.sqrt(desired_grid[0] ** 2 + desired_grid[1] ** 2)
    desired_az = _np.arctan2(desired_grid[1], desired_grid[0])
    # Convert desired grid to indices
    az_idx = ((desired_az - az_min) / _np.double(az_step))
    r_idx = ((desired_r - r_min) / _np.double(r_step))
    # clip the elments outisde of range and azimuth
    r_idx = _np.clip(r_idx, 0, image.shape[0] - 1)
    az_idx = _np.clip(az_idx, 0, image.shape[1] - 1)
    az_idx = az_idx.astype(_np.float)
    r_idx = r_idx.astype(_np.float)
    # Create interpolation function for y

    gc = _ndim.map_coordinates(image, _np.vstack((r_idx.flatten(), az_idx.flatten())), mode='constant', cval=0, order=1,
                               prefilter=False).reshape(r_idx.shape)
    # gc = bilinear_interpolate(image, az_idx, r_idx)
    gc[az_idx.astype(_np.long) == image.shape[1] - 1] = _np.nan
    gc[r_idx.astype(_np.long) == image.shape[0] - 1] = _np.nan
    gc[az_idx.astype(_np.long) == 0] = _np.nan
    gc[r_idx.astype(_np.long) == 0] = _np.nan
    LUT = r_idx + 1j * az_idx
    # Compute inverse LUT (radar coordinate -> image coordinates)
    rr, zz = _np.meshgrid(r_vec, az_vec, indexing='ij')
    xx = (rr * _np.cos(zz))
    yy = (rr * _np.sin(zz))
    # xx and yy contain Coordinates in image space (in meters!) that correspond to each radar pixel
    # we need to convert it into pixels
    xx = (xx - xx.min()) / pixel_size
    yy = (yy - yy.min()) / pixel_size
    return gc, x_vec, y_vec, LUT, xx + 1j * yy


def shadow_map(slope_angle, inc_angle):
    opp_slope_angle = slope_angle - _np.pi / 2
    look_angle = _np.pi * 2 - inc_angle
    sh_map = 0 * inc_angle.astype(_np.int8)
    sh_map[slope_angle > look_angle] += 2
    sh_map[slope_angle < inc_angle] += 7
    sh_map[opp_slope_angle > look_angle] += 8
    sh_map[opp_slope_angle > -inc_angle] += 16
    return sh_map.astype(_np.int8)


def resample_DEM(DEM, new_posting):
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
    from osgeo import gdal

    geo_t = DEM.GetGeoTransform()

    x_size = DEM.RasterXSize * _np.int(_np.abs(geo_t[1] / new_posting[0]))  # Raster xsize
    y_size = DEM.RasterYSize * _np.int(_np.abs(geo_t[5] / new_posting[1]))  # Raster ysize

    # Compute new geotransform
    new_geo = (geo_t[0], new_posting[0], geo_t[2],
               geo_t[3], geo_t[4], new_posting[1])
    mem_drv = gdal.GetDriverByName('MEM')
    dest = mem_drv.Create('', x_size, y_size, DEM.RasterCount,
                          numeric_dt_to_gdal_dt(DEM.GetRasterBand(1).DataType))
    dest.SetGeoTransform(new_geo)
    dest.SetProjection(DEM.GetProjection())
    res = gdal.ReprojectImage(DEM, dest,
                              DEM.GetProjection(), DEM.GetProjection(),
                              gdal.GRA_Bilinear)
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
    from osgeo import gdal

    x_size = arr.shape[1]
    y_size = arr.shape[0]
    # Determine number of bands
    if arr.ndim > 2:
        nbands = arr.shape[2]
    else:
        nbands = 1
    # Create a memory driver
    mem_drv = gdal.GetDriverByName('MEM')
    dest = mem_drv.Create('', x_size,
                          y_size, nbands,
                          numeric_dt_to_gdal_dt(numpy_dt_to_numeric_dt(arr.dtype.name)))
    # Set geotransform and projection
    dest.SetGeoTransform(GT)
    dest.SetProjection(proj)
    if nbands > 1:
        for band_number in range(nbands):
            dest.GetRasterBand(band_number + 1).WriteArray(arr[:, :, band_number])
    else:
        dest.GetRasterBand(1).WriteArray(arr)
    return dest


def gc_map_mask(ds_shape, lut):
    """
    Return a valid pixel map using the dataset information and the lookuptable produced by
    gc_map. Use to mask geocoded product, so that only the valid pixels (DEM pixels covered by data) are shown.
    Parameters
    ----------
    DS : iterable
        contains the shape of the dataset to mask
    lut : numpy.ndarray

    Returns
    -------
    ndar

    """
    lut_out = _np.zeros(lut.shape, dtype=bool) + 0
    lut_out[lut.imag >= ds_shape[1]] = 1
    lut_out[lut.real >= ds_shape[0]] = 1
    lut_out[(lut.imag == 0) * (lut.real == 0)] = 1
    return lut_out


def lut_lookup(LUT, radar_coord):
    """
    Return the DEM coordinates of a point in radar coordinates
    using the given LUT
    Parameters
    ----------
    LUT
    radar_coord

    Returns
    -------

    """


def paletted_to_rgb(gt):
    """
    This function converts a paletted
    geotiff to a RGB geotiff
    """
    from osgeo import gdal

    ct = gt.GetRasterBand(1).GetColorTable()
    ct.GetColorEntry(3)
    # Get raster
    RAS = gt.ReadAsArray()
    RGBA_ras = _np.zeros(RAS.shape + (4, 0))
    # Create vector of values:
    palette_list = _np.unique(RAS)
    rgba = _np.zeros((RAS.max() + 1, 4))

    for idx in palette_list:
        color = _np.array(ct.GetColorEntry(int(idx)))
        rgba[idx, :] = color
    RGBA_ras = rgba[RAS]
    mem_drv = gdal.GetDriverByName('MEM')
    output_dataset = mem_drv.Create('', RAS.shape[1], RAS.shape[0], 4, gdal.GDT_Float32)
    output_dataset = copy_and_modify_gt(RGBA_ras, gt)
    return output_dataset


def dict_to_gdal(par):
    """
    Converts dem_parameters to
    a gdal dataset
    Parameters
    ----------
    par

    Returns
    -------

    """
    #FIXME only works for swiss coordinates
    #Create raster
    (cols ,rows) = _gpf.get_shape(par)
    dt = _gpf.get_dtype(par)
    driver = gdal.GetDriverByName('MEM')
    outRaster = driver.Create('', cols, rows, 1,
                              numeric_dt_to_gdal_dt(gdalnumeric.NumericTypeCodeToGDALTypeCode(dt.type)))
    #Set coordinate system
    srs = _osr.SpatialReference()
    srs.ImportFromEPSG(21781)
    outRaster.SetProjection(srs.ExportToWkt())
    outRaster.SetGeoTransform(get_geotransform(par))
    return outRaster


def gdal_to_dict(ds):
    # Mapping from wkt to parameters
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
    # Srs from wkt
    wkt = ds.GetProjection()
    srs = _osr.SpatialReference(wkt=wkt)
    #Get projection info
    proj_arr = srs.ExportToPCI()
    # Array for the ellipsoid
    ell_arr = proj_arr[2]
    # Create new dict
    proj_dict = _coll.OrderedDict()
    # Part 1: General Parameters
    proj_dict['title'] = {'value': 'DEM'}
    proj_dict['DEM_projection'] = {'value': 'OMCH'}
    # Set the type according to the dem type
    tp = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
    if tp == 'Float32':
        proj_dict['data_format'] = {'value': 'REAL*4'}
    if tp == 'Int32':
        proj_dict['data_format'] = {'value': 'INTEGER*2'}
    if tp == 'UInt16':
        proj_dict['data_format'] = {'value': 'SHORT INTEGER'}
    proj_dict['DEM_hgt_offset'] = {'value': 0}
    proj_dict['DEM_scale'] = {'value': 1.0}
    proj_dict['width'] = {'value': ds.RasterXSize}
    proj_dict['nlines'] = {'value': ds.RasterYSize}
    gt = ds.GetGeoTransform()
    proj_dict['corner_east'] = {'value': gt[0], 'unit': 'm'}
    proj_dict['corner_north'] = {'value': gt[3], 'unit': 'm'}
    proj_dict['post_north'] = {'value': gt[5], 'unit': 'm'}
    proj_dict['post_east'] = {'value': gt[1], 'unit': 'm'}
    # TODO allow using other ellipsods
    # Part 2: Ellipsoid Parameters
    proj_dict['ellipsoid_name'] = {'value': 'Bessel 1841'}
    proj_dict['ellipsoid_ra'] = {'value': srs.GetSemiMajor(), 'unit': 'm'}
    rf = srs.GetSemiMajor() / (srs.GetSemiMajor() - srs.GetSemiMinor())
    proj_dict['ellipsoid_reciprocal_flattening'] = {'value': rf}
    # TODO allow using other datums
    # Part 3: Datum Parameters
    datum_info =srs.GetTOWGS84()
    proj_dict['datum_name'] = {'value': 'SWiss National 7PAR'}
    proj_dict['datum_country_list'] = {'value': 'Switzerland'}
    proj_dict['datum_shift_dx'] = {'value': datum_info[0], 'unit': 'm'}
    proj_dict['datum_shift_dy'] = {'value': datum_info[1], 'unit': 'm'}
    proj_dict['datum_shift_dz'] = {'value': datum_info[2], 'unit': 'm'}
    proj_dict['datum_scale_m'] = {'value': 0.0, 'unit': 'm'}
    proj_dict['datum_rotation_alpha'] = {'value': 0.0, 'unit': 'arc-sec'}
    proj_dict['datum_rotation_beta'] = {'value': 0.0, 'unit': 'arc-sec'}
    proj_dict['datum_rotation_gamma'] = {'value': 0.0, 'unit': 'arc-sec'}
    # Part 4: Projection Parameters for UTM, TM, OMCH, LCC, PS, PC, AEAC, LCC2, OM, HOM coordinates
    proj_dict['projection_name'] = {'value': 'OM - Switzerland'}
    proj_dict['projection_zone'] = {'value': 0}
    if proj_dict['DEM_projection']['value'] in ['UTM', "TM", "OMCH", "LCC", "PS", "PC", "AEAC", "LCC2", "OM", "HOM"]:
        proj_dict['center_latitude'] = {'value': srs.GetProjParm('latitude_of_center')}
        proj_dict['center_longitude'] = {'value': srs.GetProjParm('longitude_of_center')}
        proj_dict['projection_k0'] = {'value':  srs.GetProjParm('scale_factor')}
        proj_dict['false_easting'] = {'value': srs.GetProjParm('false_easting')}
        proj_dict['false_northing'] = {'value': srs.GetProjParm('false_northing')}
    proj_dict['file_title'] = {'value': 'Gamma DIFF&GEO DEM/MAP parameter file\n'}
    proj_dict = _params.ParameterFile.from_dict(proj_dict)
    return proj_dict


def geotif_to_dem(gt, par_path, bin_path):
    """
    This function converts a gdal dataset
    DEM into a gamma format pair
    of binary DEM and parameter file
    """
    # Open the data set
    try:
        DS = gdal.Open(gt)
    except RuntimeError:
        DS = gt
    # Convert
    dem_dic = gdal_to_dict(DS)
    _gpf.dict_to_par(dem_dic, par_path)
    dem = DS.GetRasterBand(1).ReadAsArray()
    dem.astype(_gpf.type_mapping[dem_dic['data_format']]).tofile(bin_path)

def extent_to_corners(ext):
    corners = []
    for x,y in zip(ext[0:2],ext[2:4]):
        corners.append([x,y])
    return corners


def get_geotransform(dem_par):
    """
    Return geotransfrom from dem_par dictionary
    Parameters
    ----------
    dem_par:
        pyrat.fileutils.parameters.ParameterFile, object

    Returns
    -------

    """
    return dem_par.corner_east, dem_par.post_east, 0, dem_par.corner_north, 0, dem_par.post_north


def get_extent(geotransform, shape):
    x = sorted((geotransform[0], geotransform[0] + geotransform[1] * shape[0]))
    y = sorted((geotransform[3], geotransform[3] + geotransform[5] * shape[1]))
    return x[0], x[1], y[0], y[1]


def get_ds_extent(ds):
    gt = ds.GetGeoTransform()
    ext = get_extent(gt, (ds.RasterXSize, ds.RasterYSize))
    return ext


def estimate_heading(mli_par, radar_coord, carto_azimuth):
    """
    Estimates the heading of GPRI data using
    the mli parameters, a reference location in radar coordinates
    and its azimuth as read from the map
    Parameters
    ----------
    mli_par
    radar_coord
    geographical_coord

    Returns
    -------

    """
    # eccess heading of the point (heading w.r.t radar image center)
    xc_heading = mli_par.GPRI_az_angle_step * (radar_coord[1] - mli_par.azimuth_lines / 2)
    return carto_azimuth - xc_heading


def geo_coord_to_dem_coord(coord, dem_par):
    """
    Convert a  geographical coordinates into
    gamma DEM coordinates
    Parameters
    ----------
    coord : array_like
        coordinates in form
    dem_par :
        gamma DEM parameters as pyrat.fileutils.parameters.ParameterFile, object
    Returns
    -------

    """
    try:
        dem_par.width
    except AttributeError:
        try:
            dem_par = _gpf.par_to_dict(dem_par)
        except FileNotFoundError:
            FileNotFoundError('The file {dem_par} does not exist'.format(dem_par=dem_par))
    x_DEM = coord[0] - dem_par.corner_east / dem_par.post_east
    y_DEM = coord[1] - dem_par.corner_north / dem_par.post_north
    return (x_DEM, y_DEM)


# TODO: only works with omch
def basemap_dict_from_gt(DS):
    """
    Creates a dict of parameters to be used with Basemap
    Parameters
    ----------
    DS

    Returns
    -------

    """
    sr = _osr.SpatialReference().ImportFromWkt(DS.GetProjection)


def clip_dataset(DS, dem_par):
    """
    Segement a geotif to correspond to the size
    specified by "dem_par"
    Parameters
    ----------
    gt
    dem_par

    Returns
    -------

    """
    try:
        dem_par.width
    except AttributeError:
        try:
            dem_par = _gpf.par_to_dict(dem_par)
        except FileNotFoundError:
            FileNotFoundError('The file {dem_par} does not exist'.format(dem_par=dem_par))
    # DS = gdal.Open(gt)
    seg_gt = get_geotransform(dem_par)
    mem_drv = gdal.GetDriverByName('MEM')
    # pixel_spacing_x = seg_gt[1]
    # pixel_spacing_y = seg_gt[5]
    dest = mem_drv.Create('', int(dem_par.width), int(dem_par.nlines), DS.RasterCount, DS.GetRasterBand(1).DataType)
    dest.SetGeoTransform(seg_gt)
    dest.SetProjection(DS.GetProjection())
    res = gdal.ReprojectImage(DS, dest,
                              DS.GetProjection(), DS.GetProjection(),
                              gdal.GRA_Bilinear)
    return dest


def rasterize_shapefile(outline, attribute_filter, x_posting=1, y_posting=1, burn_value=255):
    """
    Rasterizes the selected features in a shapefile, selected using `attribute_filter
    and returns a memory driver object
    Parameters
    ----------
    shapefile
    input_raster
    output_raster

    Returns
    -------

    """
    outline_layer = outline.GetLayer()
    outline_layer.SetAttributeFilter(attribute_filter)

    x_min, x_max, y_min, y_max = outline_layer.GetExtent()
    raster_x = int(abs(x_max - x_min) // x_posting)
    raster_y = int(abs(y_max - y_min) // y_posting)
    gt = [x_min, x_posting, 0, y_min, 0,y_posting]

    #Set the out raster
    goal_raster = gdal.GetDriverByName('MEM').Create('a',raster_x, raster_y, 1, gdal.GDT_Byte)
    goal_raster.SetGeoTransform(gt)
    goal_raster.SetProjection(outline_layer.GetSpatialRef().ExportToWkt())
    band = goal_raster.GetRasterBand(1)
    #write the shapefile
    band.Fill(0)
    gdal.RasterizeLayer(goal_raster,[1], outline_layer, burn_values=[burn_value] )
    return goal_raster


def interpolate_complex(data, LUT, **kwargs):
    """
    Returns a function to interpolate
    a complex dataset if the input data is complex,
    the normal scipy.ndimage.map_coordinates otherwise
    Parameters
    ----------
    args
    kwargs

    Returns
    -------

    """
    if _np.iscomplexobj(data):
        data_interp = _ndim.map_coordinates(data.real, LUT, **kwargs) + 1j * _ndim.map_coordinates(data.imag, LUT,
                                                                                                   **kwargs)
    else:
        data_interp = _ndim.map_coordinates(data, LUT, **kwargs)
    return data_interp


def get_reference_coord(dict, reference_name):
    reference_feature = [f for f in dict['features'] if f['id'] == reference_name]
    radar_coord = reference_feature[0]['properties']['radar_coordinates']
    return radar_coord


class GeocodingTable(object):
    """
    Class to represent geocoding tables (
    """

    def __init__(self, dem_par, lut, mli_par, inverse_lut):
        lut, dem_par = _gpf.load_dataset(dem_par, lut, dtype=_gpf.type_mapping["FCOMPLEX"])
        inverse_lut, mli_par = _gpf.load_dataset(mli_par, inverse_lut, dtype=_gpf.type_mapping["FCOMPLEX"])
        #Setup transforms
        self.dem2radar = _transf.ComplexLut(lut)
        self.radar2dem = _transf.ComplexLut(inverse_lut)
        self.gt = _transf.GeoTransform(get_geotransform(dem_par))
        self.dem_idx_to_geo_t = self.gt
        self.geo_to_dem_idx_t = self.gt.inverted()
        self.params = dem_par.copy()

    @classmethod
    def fromfile(cls, dem_par, lut, mli_par, inverse_lut ):
        return cls.__init__(dem_par, lut, mli_par, inverse_lut)

    def __getitem__(self, item):
        return self.lut.__getitem__(item)


    def dem_idx_to_radar_idx(self, dem_index):
        return self.dem2radar.transform(dem_index)

    def radar_idx_to_dem_idx(self, radar_index):
        return self.radar2dem.transform(radar_index)

    def geo_coord_to_dem_coord(self, coord):
        return self.geo_to_dem_idx_t.transform(coord)

    def dem_coord_to_geo_coord(self, coord):
        return self.gt.transform(coord)

    def geo_coord_to_radar_coord(self, geo_coord):
        t =  self.geo_to_dem_idx_t + self.dem2radar
        # dem_coord = self.geo_coord_to_dem_coord(geo_coord)
        # coord = self[int(dem_coord[0]), int(dem_coord[1])]
        return t.transform(geo_coord)

    def radar_coord_to_dem_coord(self, coord):
        return self.dem_idx_to_radar_idx(coord)

    def radar_coord_to_geo_coord(self, coord):
        return self.gt.transform(self.radar2dem.transform(coord))


    def get_extent(self):
        return get_extent(self.geotransform, [self.params.width,self.params.nlines])

    def get_geocoded_extent(self, data):
        top_left = [0, 0]
        bottom_left =  [0, data.shape[1]]
        top_middle = [0,  data.shape[1]//2]
        top_right = [data.shape[0],0]
        bottom_middle = [data.shape[0], data.shape[1]//2]
        bottom_right = [data.shape[0], data.shape[1]]
        ext_vec = _np.vstack([self.radar_coord_to_geo_coord(coord) for coord in [top_left,top_right,top_middle, bottom_right, bottom_middle, bottom_left]])
        return [ext_vec[:,0].min(), ext_vec[:,0].max(),ext_vec[:,1].min(), ext_vec[:,1].max()]

    def geocode_data(self, data):
        gc_data =  self.dem2radar.transform_array(data)
        return data.__array_wrap__(gc_data)


    @property
    def geotransform(self):
        return get_geotransform(self.params)

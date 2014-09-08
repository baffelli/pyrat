# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:32:36 2014

@author: baffelli
"""
import numpy as np
import matplotlib
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt
import pyrat
import pyrat.core.polfun
import mayavi
from mayavi import mlab

def compute_dim(WIDTH,FACTOR):
    """
    This function computes the figure size
    given the latex column width and the desired factor
    """
    fig_width_pt  = WIDTH * FACTOR
    
    inches_per_pt = 1.0 / 72.27
    golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
    
    fig_width_in  = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio   # figure height in inches
    fig_dims      = [fig_width_in, fig_height_in] # fig dims as a list
    return fig_dims
    


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
    top : double
        The maximum value of the scaled array.
    bottom : dobule
        The minium value of the scaled array.
    Returns
    -------
    ndarray
        The rescaled array.
    """
    data = args[0]
    if 'min_val' in kwargs:
        minVal = kwargs.get('min_val')
    else:
        minVal = np.nanmin(data)
    if 'max_val' in kwargs:
        maxVal = kwargs.get('max_val')
    else:
        maxVal = np.nanmax(data)
    if 'top' in kwargs:
        topV = kwargs.get('top')
    else:
        topV = 1
    if 'bottom' in kwargs:
        bottomV = kwargs.get('bottom')
    else:
        bottomV = 0
    scaled = (topV - bottomV) * ((data - minVal)) /(maxVal - minVal) + bottomV
    return scaled


def copy_and_modify_gt(RAS,gt):
    from osgeo import gdal
    mem_drv = gdal.GetDriverByName( 'MEM' )
    output_dataset = mem_drv.Create('', RAS.shape[1], RAS.shape[0], RAS.shape[2], gdal.GDT_Float32)  
    output_dataset.SetProjection(gt.GetProjection())  
    output_dataset.SetGeoTransform(gt.GetGeoTransform())  
    for n_band in range(RAS.shape[-1] - 1):
        output_dataset.GetRasterBand(n_band + 1).WriteArray( RAS[:,:,n_band] )
    return output_dataset

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
    x_size = gt_to_project.RasterXSize # Raster xsize
    y_size = gt_to_project.RasterYSize # Raster ysize
    (ulx, uly, ulz ) = tx.TransformPoint( geo_t[0], geo_t[3])
    (lrx, lry, lrz ) = tx.TransformPoint( geo_t[0] + geo_t[1]*x_size, \
                                          geo_t[3] + geo_t[5]*y_size )
     #Compute new geotransform
    new_geo = ( ulx, geo_t_ref[1], geo_t[2], \
                uly, geo_t[4], geo_t_ref[5] )
    mem_drv = gdal.GetDriverByName( 'MEM' )
    pixel_spacing_x = geo_t_ref[1]
    pixel_spacing_y = geo_t_ref[5]
    dest = mem_drv.Create('', int((lrx - ulx)/pixel_spacing_x), \
            int((uly - lry)/np.abs(pixel_spacing_y)), gt_to_project.RasterCount, gdal.GDT_Float32)
    print type(dest)
    dest.SetGeoTransform( new_geo )
    dest.SetProjection(gt_reference.GetProjection())
    res = gdal.ReprojectImage( gt_to_project, dest, \
                gt_to_project.GetProjection(), gt_reference.GetProjection(), \
                gdal.GRA_Bilinear )
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
    RGBA_ras = np.zeros(RAS.shape + (4,0))
    #Create vector of values:
    palette_list = np.unique(RAS)
    rgba = np.zeros((RAS.max() + 1 , 4))
    print rgba.shape
    for idx in palette_list:
        color = np.array(ct.GetColorEntry(int(idx))) / 255.0
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
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = cdf / cdf[-1] #normalize
    #use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape)

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
    return scale_array(1 - np.exp(- k * np.abs(im)))

def ground_to_slant(DEM,center,r_vec,az_vec,heading, interp = None):
    """
    This function transforms a ground range image
    into a slant range image
    """
    r, th = np.meshgrid(r_vec,az_vec + heading)
    #Start position
    x_0 = center[0] 
    y_0 = center[1]
    z_0 = read_coordinate_extent(DEM,(x_0,y_0), interp = interp)
    #Read DEM in pseudo ground coordinates
    x_1 = x_0 + r * np.cos(th)
    y_1 = y_0 + r * np.sin(th)
    z_1 = read_coordinate_extent(DEM,(x_1,y_1), interp = interp)
    #Incidence angle
    phi = np.arcsin((z_1 - z_0) / r)
    #Ground range
    r_gr = r * np.cos(phi)
    #Grid positions   
    x1 = x_0 + r_gr * np.cos(th)
    y1 = y_0 + r_gr * np.sin(th)
    D1 =  read_coordinate_extent(DEM,(x1,y1), interp = interp)
    return D1, x1, y1, phi
    
    
def bilinear_interpolate(im, x, y):
    
        
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

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
    
    return wa*Ia + wb*Ib + wc*Ic + wd*Id
    

def cart_to_pol(ds,center, r_vec, az_vec, heading, interp = None):
    """
    This function converts a DS object into a 
    polar coordinate scan
    """
    r, th = np.meshgrid(r_vec,az_vec + heading)
    x = r * np.cos(th)
    y = r * np.sin(th)
    x = center[0] + x
    y = center[1] + y
    image_1 = read_coordinate_extent(ds,(x,y), interp = interp)
    return image_1, x, y
    
def pol_to_cart(image,center,r_vec,az_vec,x_vec,y_vec):
    x, y = np.meshgrid(x_vec - center[0],y_vec - center[1])
    r_post =  r_vec[1] - r_vec[0]
    az_post = az_vec[1] - az_vec[0]
    r = np.sqrt(x**2 + y**2)
    th = np.arctan2(y,x)
    r_idx = r / r_post
    th_idx = th / az_post
    image_1 = bilinear_interpolate(image, r_idx, th_idx)
    return image_1, r_idx, th_idx

def get_extent(ds):
    gt = ds.GetGeoTransform()
    gt_x_vec = (gt[0], gt[0] + gt[1] * (ds.RasterXSize ))
    gt_y_vec = (gt[3], gt[3] + gt[5] * (ds.RasterYSize))
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
            px = np.int(px)
            py = np.int(py)
        else:
            px = np.clip((px),0,RAS.shape[1] -1).astype(np.int)
            py = np.clip((py),0,RAS.shape[0] -1).astype(np.int)
        return RAS[py,px]
    else:
        return interp(RAS,px,py)
    
    
    
def simulate_acquisition(DEM,center,r_vec,az_vec, heading):
    """
    Parameters
    ----------
        DEM : osgeo.gdal.Dataset
        The dem from which to compute the masks
    center : tuple
        The center in map coordinates
    az_vec  : ndarray
        The array of angles to scan
    r_vec : ndarray
        The array of distances to scan
    heading : double
        The heading of the scan
    """
    #Acquire DEM in Radar Coordinates
    DEM_pol,x ,y = cart_to_pol(DEM,center, r_vec, az_vec, heading, interp = bilinear_interpolate)
    #Compute vector position of each point in scanned DEM
    positions = np.dstack((x, y, DEM_pol))
    a = positions - np.roll(positions,1,axis = 0)
    b = positions - np.roll(positions,1,axis = 1)
    c = positions + np.roll(positions,-1,axis = 0)
    d = positions + np.roll(positions,-1,axis =1)
    normal = np.cross(a,b) + np.cross(c,d) /2 
    normal = normal / (np.linalg.norm(normal,axis =2))[:,:,None]
    #Compute vectors
    x_vec = center[0] - x 
    y_vec = center[1] - y
    z_vec = read_coordinate_extent(DEM,center) - DEM_pol
    r_vec = np.dstack((x_vec,y_vec,z_vec))
    pos = np.dstack((x,y,DEM_pol))
    alpha = np.einsum('...i,...i',normal,r_vec) / np.linalg.norm(r_vec,axis =2)
    return pos, r_vec,normal,alpha
    
def layover_and_shadow(DEM,center,az_vec,r_vec):
    """
    Parameters
    ----------
    DEM : osgeo.gdal.Dataset
        The dem from which to compute the masks
    center : tuple
        The center in map coordinates
    az_vec  : ndarray
        The array of angles to scan
    r_vec : ndarray
        The array of distances to scan
    heading : double
        The heading of the scan
    """
    
    start_coord = (center[0], center[1], DEM[center[0], center[1]])
    
    r = np.arange(image.shape[1])
    z = image
    d = np.sqrt((z[az_center,0] - z)**2 + r[None,:]**2)
    l_map = np.zeros_like(image)
    sh_map = np.zeros_like(image)
    #For each range line
    max_r = d[:,0]
    max_r_1 = max_r * 1
    for idx_r in range(d.shape[1]):   
        next_r = d[:,idx_r]
        #Compute layover map
        select_arr = max_r > next_r
        max_r[select_arr] = next_r[select_arr]
        l_map[:,idx_r] = select_arr
        #Compute shadow map
        #if the next distance is smaller than the current
        #then we have shadow
        shadow = next_r < max_r_1
        max_r_1[~shadow] = next_r[~shadow]
        sh_map[:,idx_r] = ~shadow
        
    return d, l_map, sh_map


#def dem_to_gpri(dem,dem_dict,gpri_dict):
    
    

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
    r_max = np.max(r_vec)
    r_min = np.min(r_vec)
    az_max = np.max(az_vec)
    az_min = np.min(az_vec)
    az_step = (az_vec[1] - az_vec[0])
    r_step = np.abs(r_vec[1] - r_vec[0])
    #Compute desired grid
    az_vec_1 = np.linspace(az_min,az_max,10)
    r_vec_1 = np.linspace(r_min,r_max,10)
    bound_grid = np.meshgrid(az_vec_1,r_vec_1)
    x = bound_grid[1] * np.cos(bound_grid[0])
    y = bound_grid[1] * np.sin(bound_grid[0])
    y_vec = (y.min(),y.max())
    x_vec = (x.min(),x.max())
    y_vec = np.arange(y_vec[0],y_vec[1],pixel_size)
    x_vec = np.arange(x_vec[0],x_vec[1],pixel_size)
    desired_grid = np.meshgrid(x_vec,y_vec,indexing ='xy')
    desired_r = np.sqrt(desired_grid[0]**2 + desired_grid[1]**2)
    desired_az = np.arctan2(desired_grid[1],desired_grid[0])
    #Convert desired grid to indices
    az_idx = ((desired_az - az_min) / np.double(az_step))
    r_idx = ((desired_r - r_min) / np.double(r_step))
    r_idx = np.clip(r_idx,0,image.shape[1]-1)
    az_idx = np.clip(az_idx,0,image.shape[0]-1)
    az_idx = az_idx.astype(np.float)
    r_idx = r_idx.astype(np.float)
    gc = bilinear_interpolate(image,r_idx,az_idx)
    gc[az_idx.astype(np.long) == image.shape[0] -1] = np.nan
    gc[r_idx.astype(np.long) == image.shape[1] - 1] = np.nan
    gc[az_idx.astype(np.long) == 0] = np.nan
    gc[r_idx.astype(np.long) == 0] = np.nan
    return gc, x_vec, y_vec
    
def pauli_rgb(scattering_vector, normalized= False, log=False, k = [1, 1,1]):
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
        data_diagonal = np.abs(scattering_vector) 
        if not normalized:
            if log:
                span = np.log10(np.sum(data_diagonal * np.array([1,2,1])[None,None,:],axis = 2))
                data_diagonal = np.log10(data_diagonal)
            else:
                R = data_diagonal[:,:,0] / np.max(data_diagonal[:,:,0])
                G = data_diagonal[:,:,1] / np.max(data_diagonal[:,:,1])
                B = data_diagonal[:,:,1] / np.max(data_diagonal[:,:,1])
                span = np.sum(data_diagonal,axis = 2)
                out = 1 - np.exp(- data_diagonal * np.array(k)[None,None, :])
                return out
            R = scale_array(data_diagonal[:,:,0], max_val =  0.99 * data_diagonal[:,:,0].max())
            G = scale_array(data_diagonal[:,:,1], max_val =  0.99 * data_diagonal[:,:,1].max())
            B = scale_array(data_diagonal[:,:,2], max_val =  0.99 * data_diagonal[:,:,0].max())
            out = np.zeros(R.shape+(3,))
            out[:,:,0] = R
            out[:,:,1] = G
            out[:,:,2] = B
        else:
            span = np.sum(scattering_vector,axis=2)
            out = np.abs(data_diagonal /span[:,:,None])
        return out
def show_geocoded(geocoded_image_list, n_ticks = 4, orientation = 'landscape',**kwargs):
        """
        This function is a wrapper to call imshow with a 
        list produced by the geocode_image function.
        ----------
        geocoded_image_list : list 
            list containing the geocoded image and the x and y vectors of the new grid.
        """
        ax = plt.gca()
        a = geocoded_image_list[0]
        if a.ndim is 3:
            alpha = np.isnan(a[:,:,0])
            a = np.dstack((a,~alpha))
        else:
            a = np.ma.masked_where(np.isnan(a), a)
        xv, yv = geocoded_image_list[1:None]
        x_min = np.floor(xv.min())
        x_max = np.ceil(xv.max())
        y_min = np.floor(yv.min())
        y_max = np.ceil(yv.max())
        if orientation is 'portrait':
            ext = [x_min, x_max, y_min, y_max]
            a1 = a
        else:
            ext = [y_min, y_max, x_min, x_max]
            if a.ndim is 2:
                a1 = a.T
            else:
                a1 = a.transpose([1,0,2])
        plt.imshow(a1, extent = ext, origin = 'lower'  ,**kwargs)
        plt.xticks(rotation=90)
        plt.xlabel('distance [m]')
        plt.ylabel('distance [m]')
#        xv_idx = np.linspace(0,len(xv)-1,n_ticks).astype(np.int)
#        yv_idx = np.linspace(0,len(yv)-1,n_ticks).astype(np.int)
#        plt.xticks(xv_idx,np.ceil(xv[xv_idx]))
#        plt.yticks(yv_idx,np.ceil(yv[yv_idx]))
        ax.set_aspect('equal')
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
        self.mask = np.zeros(self.shape,dtype=np.bool)
        self.fill_mask()

        
    def fill_mask(self):
        def dstack_product(x, y):
            return np.dstack(np.meshgrid(x, y)).reshape(-1, 2)
        x_grid = np.arange(self.shape[0])
        y_grid = np.arange(self.shape[1])
        points = dstack_product(y_grid, x_grid)
        for pt in self.polygon_list:
            path = matplotlib.path.Path(pt)
            self.mask = self.mask + path.contains_points(points, radius = 0.5).reshape(self.shape)
            
    def draw_mask(self,ax,**kwargs):
        display_to_ax = ax.transAxes.inverted().transform
        data_to_display = ax.transData.transform
        for dta_pts in self.polygon_list:
            ax_pts = display_to_ax(data_to_display(dta_pts))
            p = plt.Polygon(ax_pts, True, transform=ax.transAxes,**kwargs)
            ax.add_patch(p)


def ROC(cases,scores,n_positive, n_negative):
    sort_indices = np.argsort(scores)
    sort_cases = cases[sort_indices[::-1]]
    pd = np.cumsum(sort_cases == 1) / np.double(n_positive)
    pf = np.cumsum(sort_cases == 0) / np.double(n_negative)
    return pf, pd


def confusionMatrix(image,masks,training_areas,function, threshold):
    #Compute classes taking the mean of the class pixels
    classes = ()
    for mask_idx, mask in enumerate(training_areas):
        classes = classes + (np.mean(image[mask],axis = 0),)
    classified, distance = pyrat.classifier(image,classes, function, threshold)
    cm_1 = np.zeros((len(masks),len(masks)))
    for idx_1,mask in enumerate(masks):
        for idx_2,mask1 in enumerate(masks):
            cm_1[idx_1,idx_2] = np.sum(classified[mask1] == idx_1 + 1) / np.double(np.sum(mask1 == True))
    return cm_1, classified, distance
    
def rectangle_vertices(v1,v2):
    x1 = v1[1]
    y1 = v1[0]
    x2 = v2[1]
    y2 = v2[0]
    return np.array([[x1,y1],[x1,y2],[x2,y2],[x2,y1]])

def scale_coherence(c):
    
        return (np.sin(c * np.pi  - np.pi / 2) + 1)/2 * (c > 0.3)

  
def if_hsv(ifgram):
     H = scale_array(np.angle(ifgram))
     S = np.ones_like(H)
     V = scale_coherence(np.abs(ifgram))
     RGB = matplotlib.colors.hsv_to_rgb(np.dstack((H, S, V)))
     return RGB

def show_if(S1, S2, win):
    name_list = ['HH', 'HV', 'VH', 'VV']
    k1 = S1.scattering_vector(basis='lexicographic')
    k2 = S2.scattering_vector(basis='lexicographic')
    if_mat = np.zeros(S1.shape[0:2] + (4,4), dtype = np.complex64)
    for i in range(4):
        for j in range(4):
            c_if = pyrat.coherence(k1[:, :, i], k2[:, :, j], win)
            if_mat[:,:,i,j] = c_if
            RGB = if_hsv(c_if)
            if i == 0 and j == 0:
                ax =  plt.subplot2grid((4, 4), (i, j))
                plt.imshow(RGB,  cmap = 'gist_rainbow', interpolation = 'none')
                ax = plt.gca()
            else:
                 plt.subplot2grid((4, 4), (i, j))
                 plt.imshow(RGB , cmap = 'gist_rainbow', interpolation = 'none')
            plt.title(name_list[i] + name_list[j])
    return if_mat
            
def hsv_cp(H,alpha,span):
    V = scale_array(np.log10(span))    
    H1 = scale_array(alpha, top = 0, bottom = 240)  / 360
    S = 1 - H
    return matplotlib.colors.hsv_to_rgb(np.dstack((H1,S,V)))

def show_signature(signature_output):
    phi = signature_output[2]
    tau = signature_output[3]
    sig_co = signature_output[0]
    sig_cross = signature_output[1]
    phi_1, tau_1 = np.meshgrid(phi, tau)
    xt = [-45,45,-90,90]
    f_co = plt.figure()
    plt.imshow(sig_co, interpolation = 'none', cmap =  'RdBu_r', extent = xt)
    plt.locator_params(nbins=5)
    plt.xlabel(r'ellipicity $\tau$')
    plt.ylabel(r'orientation $\phi$')
    plt.xticks(rotation=90)

#    plt.axis('equal')
    f_x = plt.figure()
    plt.imshow(sig_cross, interpolation = 'none', cmap =  'RdBu_r', extent = xt)
    plt.locator_params(nbins=5)
    plt.xlabel(r'ellipicity $\tau$')
    plt.ylabel(r'orientation $\phi$')
    plt.xticks(rotation=90)

#    plt.axis('equal')
    return f_co, f_x


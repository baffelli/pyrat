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
    az_step = np.abs(az_vec[1] - az_vec[0])
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

    #%Try 1d interpolation
#    x = np.arange(np.prod(image.shape[0:2]))
#    y = image.flatten()
#    desired_coordinates = az_idx.flatten() * image.shape[1] + r_idx.flatten()
    gc = bilinear_interpolate(image,r_idx,az_idx)
#    if image.ndim is 2:
##        gc = np.interp(desired_coordinates,x,y,left = np.nan,right = np.nan).reshape(r_idx.shape)
#            gc = bilinear_interpolate(image,r_idx,az_idx)
#    if image.ndim is 3:
#        chans = np.dsplit(image,image.shape[2])
#        new_chans = []
#        for chan in chans:
#            print len(chan)
#            print len(x)
##            gc_chan = np.interp(desired_coordinates,x,chan.squeeze().flatten(),left = np.nan,right = np.nan).reshape(r_idx.shape)
#            gc_chan = bilinear_interpolate(chan,r_idx,az_idx)
#            new_chans = new_chans + [gc_chan,]
#        gc = np.dstack(new_chans)
            
#    nd = image.ndim
#    if nd is 2:
#        idx_vec = (az_idx,r_idx)
#    elif nd > 2:
#        remaining_dim = image.ndim - 2
#        remaining_axes = (Ellipsis,) * remaining_dim
#        idx_vec = (az_idx,r_idx) + remaining_axes
#    #Take care of points outside of the image
#    gc = image[idx_vec]
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
def show_geocoded(geocoded_image_list, n_ticks = 4,**kwargs):
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
#        a[np.isnan(a)] = 0
        plt.imshow(a,**kwargs)

        xv, yv = geocoded_image_list[1:None]
        xv_idx = np.linspace(0,len(xv)-1,n_ticks).astype(np.int)
        yv_idx = np.linspace(0,len(yv)-1,n_ticks).astype(np.int)
        plt.xticks(xv_idx,np.ceil(xv[xv_idx]))
        plt.yticks(yv_idx,np.ceil(yv[yv_idx]))
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
    f_co = mlab.figure()
    mlab.mesh(phi_1,tau_1,np.abs(sig_co).transpose(), representation = 'wireframe')
    f_x = mlab.figure()
    mlab.mesh(phi_1,tau_1,np.abs(sig_cross).transpose(), representation = 'wireframe')
    return f_co, f_x


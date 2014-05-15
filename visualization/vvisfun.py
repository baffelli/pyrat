# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:32:36 2014

@author: baffelli
"""

def scale_array(*args,**kwargs):
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

def sigmoid_stretch(data,alpha):
    return 1/(1+np.exp((-data/alpha)/(np.nanmax(data)-np.nanmin(data))))
    
def histeq(im,nbr_bins=256):
   #get image histogram
   imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = cdf / cdf[-1] #normalize

   #use linear interpolation of cdf to find new pixel values
   im2 = interp(im.flatten(),bins[:-1],cdf)
   return im2.reshape(im.shape)
    
def geocode_image(*args):
    image = args[0]
    pixel_size = args[1]
    try:
        r_vec = image.r_vec
        az_vec = image.az_vec
    except AttributeError:
        if len(args) >= 4:
            r_vec = args[3]
            az_vec = args[2]
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
    bound_grid = np.meshgrid(az_vec,r_vec)
    x = bound_grid[1] * cos(bound_grid[0])
    y = bound_grid[1] * sin(bound_grid[0])
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    y_vec = np.sort((y_min,y_max))
    x_vec = np.sort((x_min,x_max))
    y_vec = np.arange(y_vec[0],y_vec[1],pixel_size)
    x_vec = np.arange(x_vec[0],x_vec[1],pixel_size)
    desired_grid = np.meshgrid(x_vec,y_vec,indexing ='xy')
    desired_r = np.sqrt(desired_grid[0]**2 + desired_grid[1]**2)
    desired_az = np.arctan2(desired_grid[1],desired_grid[0])
    #Convert desired grid to indices
    az_idx = np.floor((desired_az - az_min) / np.double(az_step))
    r_idx = np.ceil((desired_r - r_min) / r_step)
    r_idx = np.clip(r_idx,0,image.shape[1]-1)
    az_idx = np.clip(az_idx,0,image.shape[0]-1)
    az_idx = az_idx.astype(np.long)
    r_idx = r_idx.astype(np.int)
    nd = image.ndim
    if nd is 2:
        idx_vec = (az_idx,r_idx)
    elif nd > 2:
        remaining_dim = image.ndim - 2
        remaining_axes = (Ellipsis,) * remaining_dim
        idx_vec = (az_idx,r_idx) + remaining_axes
    #Take care of points outside of the image
    gc = image[idx_vec]
    gc[az_idx == image.shape[0] -1] = nan
    gc[r_idx == image.shape[1] - 1] = nan
    gc[az_idx == 0] =nan
    gc[r_idx == 0] = nan
    return gc, x_vec, y_vec
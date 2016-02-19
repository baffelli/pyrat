# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:32:36 2014

@author: baffelli
"""
import os as _os

import cv2 as _cv2
import matplotlib as _mpl
import matplotlib.pyplot as _plt
import numpy as _np
#import pyrat.core.polfun
import scipy.fftpack as _fftp
import scipy.ndimage as _ndim




# Define colormaps
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
        (75, 185, 79),
        (120, 196, 78),
        (177, 213, 63),
        (244, 238, 15),
        (253, 186, 47),
        (236, 85, 42),
        (222, 30, 61)
    )
    color_list = _np.array(color_list) / 255.0
    cm = _mpl.colors.ListedColormap(color_list, name='defo')
    norm = _mpl.colors.BoundaryNorm(bounds, cm.N)
    return cm, norm


def draw_shapefile(path, basename="VEC200"):
    # Create driver
    layers = ['Building',
              'FlowingWater',
              'LandCover',
              'Lake',
              'Road']
    for l in layers:
        layer_name = path + basename + '_' + l + '.shp'
        dataSource = pyrat.other_files.load_shapefile(layer_name)


def compute_dim(WIDTH, FACTOR):
    """
    This function computes the figure size
    given the latex column width and the desired factor
    """
    fig_width_pt = WIDTH * FACTOR

    inches_per_pt = 1.0 / 72.27
    golden_ratio = (_np.sqrt(5) - 1.0) / 2.0  # because it looks good

    fig_width_in = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio  # figure height in inches
    fig_dims = [fig_width_in, fig_height_in]  # fig dims as a list
    return fig_dims


def set_figure_size(f, x_size, ratio):
    y_size = x_size * ratio
    f.set_size_inches(x_size, y_size, forward=True)


def scale_array(*args, **kwargs):
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
    scaled = (_np.clip(data, minVal, maxVal) - minVal) / (maxVal - minVal)
    return scaled


def segment_image(S, thresh):
    return (S > thresh).astype(_np.bool)


def coarse_coregistration(master, slave, sl):
    # Determine coarse shift
    T = _np.abs(master[sl + (0, 0)]).astype(_np.float32)
    I = _np.abs(slave[:, :, 0, 0]).astype(_np.float32)
    T[_np.isnan(T)] = 0
    I[_np.isnan(I)] = 0
    res = _cv2.matchTemplate(I, T, _cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = _cv2.minMaxLoc(res)
    sh = (sl[0].start - max_loc[1], sl[1].start - max_loc[0])
    print sh
    slave_coarse = shift_image(slave, (-sh[0], -sh[1]))
    return slave_coarse, res


def correct_shift_radar_coordinates(slave, master, axes=(0, 1), oversampling=(5, 2), sl=None):
    import pyrat.gpri_utils.calibration as calibration

    master_1 = master.__copy__()
    if sl is None:
        M = _np.abs(master_1[:, :, 0, 0]).astype(_np.float32)
        S = _np.abs(slave[:, :, 0, 0]).astype(_np.float32)
    else:
        M = _np.abs(master_1[sl + (0, 0)]).astype(_np.float32)
        S = _np.abs(slave[sl + (0, 0)]).astype(_np.float32)
    # Get shift
    co_sh, corr = calibration.get_shift(M, S, oversampling=oversampling, axes=(0, 1))
    print(co_sh)
    slave_1 = shift_image(slave, co_sh)
    return slave_1, corr


def shift_image(image, shift):
    x = _np.arange(image.shape[0]) + shift[0]
    y = _np.arange(image.shape[1]) + shift[1]
    x, y = _np.meshgrid(x, y, indexing='xy')
    image_1 = image.__array_wrap__( \
        bilinear_interpolate(_np.array(image), y.T, x.T))
    image_1[_np.isnan(image_1)] = 0
    return image_1


def shift_image_FT(image, shift):
    # Pad at least three times the shift required
    edge_pad_size = zip([0] * image.ndim, [0] * image.ndim)
    axes = range(len(shift))
    for ax in axes:
        ps = abs(int(shift[ax])) * 0
        edge_pad_size[ax] = (ps, ps)
    image_pad = _np.pad(image, edge_pad_size, mode='constant')
    # Transform in fourier domain
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
    x = _np.linspace(0, image.shape[0], num=image.shape[0] * sampling_factor[0])
    y = _np.linspace(0, image.shape[1], num=image.shape[1] * sampling_factor[1])
    x, y = _np.meshgrid(x, y, order='xy')
    image_1 = image.__array_wrap__( \
        bilinear_interpolate(_np.array(image), y.T, x.T))
    image_1[_np.isnan(image_1)] = 0
    return image_1


def histeq(im, nbr_bins=256):
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
    # get image histogram
    imhist, bins = _np.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = cdf / cdf[-1]  # normalize
    # use linear interpolation of cdf to find new pixel values
    im2 = _np.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape)


def stretch_contrast(im, tv=5, ma=95):
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


def reverse_lookup(image, lut):
    idx_az = _np.arange(image.shape[0])
    idx_r = _np.arange(image.shape[1])
    idx_az, idx_r = _np.meshgrid(idx_az, idx_r)


def bilinear_interpolate(im, x, y):
    x = _np.asarray(x)
    y = _np.asarray(y)

    x0 = _np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = _np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = _np.clip(x0, 0, im.shape[1] - 1)
    x1 = _np.clip(x1, 0, im.shape[1] - 1)
    y0 = _np.clip(y0, 0, im.shape[0] - 1)
    y1 = _np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    if im.ndim > 2:
        trailing_dim = im.ndim - 2
        access_vector = (Ellipsis, Ellipsis) + trailing_dim * (None,)
        wa = wa[access_vector]
        wb = wb[access_vector]
        wc = wc[access_vector]
        wd = wd[access_vector]
    interp = wa * Ia + wb * Ib + wc * Ic + wd * Id
    interp[y.astype(_np.long) >= im.shape[0] - 1] = _np.nan
    interp[x.astype(_np.long) >= im.shape[1] - 1] = _np.nan
    interp[y.astype(_np.long) <= 0] = _np.nan
    interp[x.astype(_np.long) <= 0] = _np.nan
    return interp


def auto_heading(S, pixel_coord, geo_coord):
    geo_heading = _np.rad2deg(_np.arctan2(geo_coord[0], geo_coord[1]))
    pixel_az = S.az_vec[pixel_coord[0]]
    return pixel_az - geo_heading


def pauli_rgb(scattering_vector, normalized=False, k=1, sf = 1):
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
        data_diagonal = _np.abs(scattering_vector)
        # sp = data_diagonal[:,:,0]
        # min_perc = _np.percentile(sp, 5)
        # max_perc = _np.percentile(sp, 99.5)
        # min_perc = _np.nanmin(sp)
        # max_perc = _np.nanmax(sp)
        # data_diagonal[:, :, 0] = scale_array(data_diagonal[:, :, 0],
        #                                      min_val=min_perc, max_val=max_perc)
        # data_diagonal[:, :, 1] = scale_array(data_diagonal[:, :, 1],
        #                                      min_val=min_perc, max_val=max_perc)
        # data_diagonal[:, :, 2] = scale_array(data_diagonal[:, :, 2],
        #                                      min_val=min_perc, max_val=max_perc)
        data_diagonal = (sf * ((data_diagonal) ** (k)))
        data_diagonal[:, :, 0] = scale_array(data_diagonal[:, :, 0])
        data_diagonal[:, :, 1] = scale_array(data_diagonal[:, :, 1])
        data_diagonal[:, :, 2] = scale_array(data_diagonal[:, :, 2])
        R = data_diagonal[:, :, 0]
        G = data_diagonal[:, :, 1]
        B = data_diagonal[:, :, 2]
        out = _np.dstack((R, G, B))
    else:
        span = _np.sum(scattering_vector, axis=2)
        out = _np.abs(scattering_vector / span[:, :, None])
    return out


def show_geocoded(geocoded_image, **kwargs):
    """
        This function is a wrapper to call imshow with a 
        list produced by the geocode_image function.
        ----------
        geocoded_image_list : list 
            list containing the geocoded image and the x and y vectors of the new grid.
        """
    ax = _plt.gca()
    a = geocoded_image
    if a.ndim is 3 and a.shape[-1] == 3:
        alpha = _np.isnan(a[:, :, 0])
        a = _np.dstack((a, ~alpha))
    else:
        a = _np.ma.masked_where(_np.isnan(a), a)
    _plt.imshow(a, **kwargs)
    _plt.xlabel(r'Easting [m]')
    _plt.ylabel(r'Northing [m]')
    return a

def ROC(cases, scores, n_positive, n_negative):
    sort_indices = _np.argsort(scores)
    sort_cases = cases[sort_indices[::-1]]
    pd = _np.cumsum(sort_cases == 1) / _np.double(n_positive)
    pf = _np.cumsum(sort_cases == 0) / _np.double(n_negative)
    return pf, pd


def confusionMatrix(image, masks, training_areas, function, threshold):
    # Compute classes taking the mean of the class pixels
    classes = ()
    for mask_idx, mask in enumerate(training_areas):
        classes = classes + (_np.mean(image[mask], axis=0),)
    classified, distance = pyrat.classifier(image, classes, function, threshold)
    cm_1 = _np.zeros((len(masks), len(masks)))
    for idx_1, mask in enumerate(masks):
        for idx_2, mask1 in enumerate(masks):
            cm_1[idx_1, idx_2] = _np.sum(classified[mask1] == idx_1 + 1) / _np.double(_np.sum(mask1 == True))
    return cm_1, classified, distance


def rectangle_vertices(v1, v2):
    x1 = v1[1]
    y1 = v1[0]
    x2 = v2[1]
    y2 = v2[0]
    return _np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])


def scale_coherence(c):
    #    c_sc = _np.select(((_np.sin(c * _np.pi / 2)), 0.3), (c > 0.2, c<0.2))
    return _np.sin(c * _np.pi / 2)



def dismph(data, min_val=-_np.pi, max_val=_np.pi, k=1, N=24):
    #palette to scale phase
    colors = (
    (0,1,1),
    (1,0,1),
    (1,1,0),
    (0,1,1)
    )
    pal = _mpl.colors.LinearSegmentedColormap.\
        from_list('subs_colors', colors, N=N)
    norm = _mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    #Extract amplitude and phase
    ang = scale_array(_np.angle(data))
    ampl = scale_array(_np.abs(data)**k)
    #Convert angle to colors
    rgb = pal(ang)
    #Extract the hsv parameters
    hsv = _mpl.colors.rgb_to_hsv(rgb[:,:,0:3])
    hsv[:,:,1] = 0.6
    #Add
    #Scale with intensity
    hsv[:,:,2] = ampl
    mask = ampl < 0.01
    hsv[mask] = 0
    #Convert back to rgb
    rgb = _mpl.colors.hsv_to_rgb(hsv)
    return rgb[:,:,:], pal, norm

def disp_mph(data, dt='amplitude', k=0.5, min_val=-_np.pi,
             max_val=_np.pi, return_pal=False, return_im=True):



    H = scale_array(_np.angle(data), min_val=min_val, max_val=max_val)
    sat = 0.75
    S = _np.zeros_like(H) + sat
    if dt == 'coherence':
        V = scale_coherence((_np.abs(data)))
    elif dt == 'amplitude':
        V = scale_array(_np.abs(data)**k)
    elif dt == 'none':
        V = scale_array(_np.abs(data))
    RGB = _mpl.colors.hsv_to_rgb(_np.dstack((H, S, V)))
    if return_im:
        H_pal = scale_array(_np.linspace(min_val, max_val, 255))
        V_pal = _np.linspace(0, 1, 255)
        HH, VV = _np.meshgrid(H_pal, V_pal)
        SS = _np.zeros_like(VV) + sat
        im = _mpl.colors.hsv_to_rgb(_np.dstack((HH, SS, VV)))
        return RGB, im
    if return_pal:
        H_pal = scale_array(_np.linspace(min_val, max_val, 255))
        S_pal = H_pal * 0 + sat
        V_pal = H_pal * 0 + 1
        pal = _mpl.colors.hsv_to_rgb(_np.dstack((H_pal, S_pal, V_pal))).squeeze()
        cmap = _mpl.colors.LinearSegmentedColormap.from_list('my_colormap', pal, 256)
        return RGB, cmap
    else:
        return RGB


def load_custom_palette():
    RGB = disp_mph(_np.exp(1j * _np.linspace(0, 2 * _np.pi, 255))).squeeze()
    cmap = _mpl.colors.LinearSegmentedColormap.from_list('my_colormap', RGB, 256)
    return cmap


def extract_section(image, center, size):
    x = center[0] + _np.arange(-int(size[0] / 2.0), int(size[0] / 2.0))
    y = center[1] + _np.arange(-int(size[0] / 2.0), int(size[0] / 2.0))
    x = _np.mod(x, image.shape[0])
    y = _np.mod(y, image.shape[1])
    xx, yy = _np.meshgrid(x, y)
    return image[xx, yy]


def show_if(S1, S2, win):
    name_list = ['HH', 'HV', 'VH', 'VV']
    k1 = S1.scattering_vector(basis='lexicographic')
    k2 = S2.scattering_vector(basis='lexicographic')
    if_mat = _np.zeros(S1.shape[0:2] + (4, 4), dtype=_np.complex64)
    for i in range(4):
        for j in range(4):
            c_if = pyrat.coherence(k1[:, :, i], k2[:, :, j], win)
            if_mat[:, :, i, j] = c_if
            RGB = if_hsv(c_if)
            if i == 0 and j == 0:
                ax = _plt.subplot2grid((4, 4), (i, j))
                _plt.imshow(RGB, cmap='gist_rainbow', interpolation='none')
                ax = _plt.gca()
            else:
                _plt.subplot2grid((4, 4), (i, j))
                _plt.imshow(RGB, cmap='gist_rainbow', interpolation='none')
            _plt.title(name_list[i] + name_list[j])
    return if_mat


def hsv_cp(H, alpha, span):
    V = scale_array(_np.log10(span))
    H1 = scale_array(alpha, top=0, bottom=240) / 360
    S = 1 - H
    return _mpl.colors.hsv_to_rgb(_np.dstack((H1, S, V)))


def show_signature(signature_output, rotate=False):
    phi = signature_output[2]
    tau = signature_output[3]
    sig_co = signature_output[0]
    sig_cross = signature_output[1]
    phi_1, tau_1 = _np.meshgrid(phi, tau)
    if rotate:
        xt = [-90, 90, -45, 45]
        sig_co = sig_co.T
        sig_cross = sig_cross.T
    else:
        xt = [-45, 45, -90, 90]
    f_co = _plt.figure()
    _plt.imshow(sig_co, interpolation='none', cmap='RdBu_r', extent=xt)
    _plt.locator_params(nbins=5)
    _plt.xlabel(r'ellipicity $\tau$')
    _plt.ylabel(r'orientation $\phi$')
    _plt.xticks(rotation=90)

    #    _plt.axis('equal')
    f_x = _plt.figure()
    _plt.imshow(sig_cross, interpolation='none', cmap='RdBu_r', extent=xt)
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
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))

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
        self.mask = _np.zeros(self.shape, dtype=_np.bool)
        self.fill_mask()

    def fill_mask(self):
        def dstack_product(x, y):
            return _np.dstack(_np.meshgrid(x, y)).reshape(-1, 2)

        x_grid = _np.arange(self.shape[0])
        y_grid = _np.arange(self.shape[1])
        points = dstack_product(y_grid, x_grid)
        for pt in self.polygon_list:
            path = _mpl.path.Path(pt)
            self.mask = self.mask + path.contains_points(points, radius=0.5).reshape(self.shape)

    def draw_mask(self, ax, **kwargs):
        display_to_ax = ax.transAxes.inverted().transform
        data_to_display = ax.transData.transform
        for dta_pts in self.polygon_list:
            ax_pts = display_to_ax(data_to_display(dta_pts))
            p = _plt.Polygon(ax_pts, True, transform=ax.transAxes, **kwargs)
            ax.add_patch(p)




# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:32:36 2014

@author: baffelli
"""
import matplotlib as _mpl
import matplotlib.colors as _col
import matplotlib.pyplot as _plt
import numpy as _np
import scipy.fftpack as _fftp
import scipy.ndimage as _ndim
from skimage import color
from scipy.special import expit as _sigma

import itertools as _iter

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
    scaled = (data - minVal) / _np.abs(maxVal - minVal)
    return scaled

#TODO This function will be removed when matplotlib 2.0 will be availabel
def format_axes(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    return ax


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
    slave_1 = shift_image(slave, co_sh)
    return slave_1, corr


def shift_image(image, shift):
    x = _np.arange(image.shape[0]) + shift[0]
    y = _np.arange(image.shape[1]) + shift[1]
    x, y = _np.meshgrid(x, y, indexing='xy')
    image_1 = image.__array_wrap__(
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
    image_1 = image.__array_wrap__(
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


def exp_im(im, k, sf, peak=False):
    """
    Converts an image to the 0-1 range using an
    exponential scaling
    
    Parameters
    ----------
    im : array_like
        The image to convert
    k  : double
        The scaling exponent
    sf : dobule
        The relative scale factor
    peak : bool
        If set to true, data is scaled relative to the peak
    Returns
    -------
    ndarray
        The scaled image.
    """
    im_pwr = _np.abs(im)**2
    if peak:#divide by the peak
        sc = _np.nanmax(im_pwr)
    else:#search reference region
        sc= _np.nanmean(im_pwr)
    im_pwr = _np.clip(sf * (im_pwr ** k) / (sc**k),0,1)
    return im_pwr


def gamma_scaling(im, k, sf):
    im_pwr = _np.abs(im)
    im_pwr = _np.clip(sf, 0, 1) * scale_array(im_pwr ** k)
    return im_pwr


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
        access_vector = (slice(None,None),)*2 + trailing_dim * (None,)
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


def pauli_rgb(scattering_vector, normalized=False, k=0.3, sf=1, peak=False, common=True):
    """
        This function produces a rgb image from a scattering vector.
        
        Parameters
        ----------
        scattering_vector : ndarray 
            the scattering vector to be represented.
        normalized : bool
            set to true for the relative rgb image, where each channel is normalized by the sum.
        k   : float
            exponent for nonlinear scaling
        sf : float
            scaling factor for nonlinear scaling
        scaling: str
            either "common" for scaling relative to the common mean, 'single' for independent scaling or 'peak' for scaling
            relative to each channels peak
        """
    if not normalized:
        data_diagonal = _np.abs(scattering_vector)
        # Compute the percentiles for all the channels
        RGB = _np.zeros(data_diagonal.shape)
        #Reference region
        #
        if common:
            RGB = exp_im(data_diagonal, k, sf, peak=peak)
        else:
            for chan in [0, 1, 2]:
                RGB[:,:,chan] = exp_im(data_diagonal[:,:,chan], k, sf, peak=peak)

        out = RGB
    else:
        span = _np.sum(scattering_vector, axis=2)
        out = _np.abs(scattering_vector / span[:, :, None])
        out = _np.ma.masked_where(_np.isnan(out), out)
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


def mask_zeros(rgb):
    mask = _np.sum(rgb, axis=-1) == 0
    mask = 1 - mask
    rgba = _np.dstack((rgb, mask))
    return rgba


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


def scale_coherence(c, threshold=0.3, slope=12):
    """
    Scaled the coherence magnitude by a sigmoid function
    Parameters
    ----------
    c : array_like
        The coherence data to scale
    threshold : float
        The threshold where the scaled value starts increasing
    slope : float
        The slope of the scaled coherence, higher values result in more abrupt thresholding

    Returns
    -------

    """
    #    c_sc = _np.select(((_np.sin(c * _np.pi / 2)), 0.3), (c > 0.2, c<0.2))
    return _sigma(slope* c - threshold * slope)


# #TODO
class gammaNormalize(_col.Normalize):
    def __init__(self, vmin=None, vmax=None, mean=None, clip=False, gamma=1.2, sf=1):
        self.mean = mean
        self.vmax = vmax
        self.vmin = vmin
        self.gamma = gamma
        self.sf = sf
        _col.Normalize.__init__(self, vmin, vmax, clip)

    def autoscale_None(self, A):
        self.vmax = _np.nanmax(A)
        self.vmin = _np.nanmin(A)
        self.mean = _np.nanmean(A)

    def __call__(self, value):
        self.autoscale_None(value)
        return ((value - self.vmin) / (self.vmax - self.vmin)) ** self.gamma


def circular_palette(N=24, repeat=False, radius=40, lum=70):
    """
    Produces an equiluminant, circular list of colors to be used as a paletted
    for phase visualization.
    Parameters
    ----------
    N : int
        Number of steps in the palette
    repeat : bool
        If set, the palette is repeated, useful for data between -90 and 90
    radius : float
        The radius of the circle in the LAB colorspace, controls the amount of saturation
    lum

    Returns
    -------
    a rgb list representing the palet

    """
    if not repeat:
        theta = _np.linspace(0, 2 * _np.pi, N)
    else:
        theta = _np.linspace(0, 2 * _np.pi, N / 2)
    a = 2 * radius * _np.cos(theta)
    b = radius * _np.sin(theta)
    L = _np.ones(a.shape) * lum #Equiluminant
    LAB = _np.dstack((L, a, b))
    rgb = color.lab2rgb(LAB[::-1, :]).squeeze()
    if repeat:
        rgb = _np.vstack((rgb, rgb))
    rgb = _mpl.colors.ListedColormap(rgb, name='circular_phase', N=N)
    return rgb


def dismph_palette(data, N=20,**kwargs):
    mli = kwargs.pop('mli',None)#
    coherence = kwargs.pop('coherence')
    if coherence is False:
        ampl_orig = _np.abs(data)
        angle_orig = _np.angle(data)
    else:
        ampl_orig = _np.abs(mli)
        angle_orig = _np.angle(data)
    ampl, phase = [_np.linspace(_np.nanmin(chan), _np.nanmax(chan), N) for chan in [ampl_orig, angle_orig]]
    aa, pp = _np.meshgrid(ampl, phase)
    rgb, pal, norm = dismph(aa * _np.exp(1j * pp), **kwargs)
    ext = [ ampl.min(), ampl.max(),phase.min(), phase.max(), ]
    return rgb, ext



def dismph(data, min_val=-_np.pi, max_val=_np.pi, k=0.5, mli=None, peak=False, N=24, sf=1, coherence_slope=12, repeat=False, coherence=False, black_background=True, coherence_threshold=0.3):
    pal = circular_palette(N, repeat=repeat)
    norm = _mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    # Extract amplitude and phase
    ang = scale_array(_np.angle(data), min_val=min_val, max_val=max_val)
    # Convert angle to colors
    rgb = pal(ang)
    # #Extract the hsv parameters
    hsv = _mpl.colors.rgb_to_hsv(rgb[:, :, 0:3])
    #Extract hue
    H = hsv[:,:,0]
    if coherence:
        S = scale_coherence(_np.abs(data),threshold=coherence_threshold, slope=coherence_slope) * hsv[:,:,1]
        V = (exp_im(_np.abs(mli), k, sf, peak=peak))
    else:
        S = hsv[:,:,1]
        V = exp_im(_np.abs(data), k, sf, peak=peak)
    # Convert back to rgb
    rgb = _mpl.colors.hsv_to_rgb(_np.dstack((H,S,V)))
    mask = _np.sum(rgb, axis=-1) == 0
    # RGBA alpha mask
    if not black_background:
        alpha_chan = (1 - mask)
        rgb = _np.dstack((rgb, alpha_chan))
    else:
        rgb[mask] = 0
    # Analyze
    return rgb, pal, norm


def hsv_cp(H, alpha, span):
    """
    Display H entropy and span as a composite
    Parameters
    ----------
    H
    alpha
    span

    Returns
    -------

    """
    V = scale_array(_np.log10(span))
    H1 = scale_array(alpha, top=0, bottom=240) / 360
    S = 1 - H
    return _mpl.colors.hsv_to_rgb(_np.dstack((H1, S, V)))


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
    f, (ax_co, ax_x) = _plt.subplots(2, sharex=True, shary=True)
    ax_co.imshow(sig_co, interpolation='none', cmap='RdBu_r', extent=xt)
    ax_x.imshow(sig_cross, interpolation='none', cmap='RdBu_r', extent=xt)
    _plt.locator_params(nbins=5)
    _plt.xlabel(r'ellipicity $\tau$')
    _plt.ylabel(r'orientation $\phi$')
    _plt.xticks(rotation=90)
    return f


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

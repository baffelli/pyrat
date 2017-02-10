import itertools as _iter

import numpy as _np
import numpy as np
from matplotlib.transforms import Transform, Affine2D, Bbox
from scipy import interpolate as _interp

import scipy.ndimage as _ndim

def interpolate_ndim(x,y,x_new, y_new, data, **kwargs):
    new_coords = _np.vstack((x_new.flatten(), y_new.flatten()))
    if _np.iscomplexobj(data):
        data_interp = _ndim.map_coordinates(data.real, new_coords, **kwargs) + 1j * _ndim.map_coordinates(data.imag, new_coords,
                                                                                                   **kwargs)
    else:
        data_interp = _ndim.map_coordinates(data, new_coords, **kwargs)
    return data_interp.reshape(x_new.shape)





class ComplexLut(Transform):
    """
    Transform from DEM-indices to radar indices
    """

    def __init__(self, lut):
        super().__init__()
        # Create interpolator
        x_vec = _np.arange(lut.shape[0])
        y_vec = _np.arange(lut.shape[1])
        self.real_interp = _interp.RectBivariateSpline(x_vec, y_vec, lut.real, kx=1, ky=1)
        self.imag_interp = _interp.RectBivariateSpline(x_vec, y_vec, lut.imag, kx=1, ky=1)
        self.lut = lut
        self.input_dims = 2
        self.output_dims = 2
        self.is_separable = False
        self.has_inverse = False

    def transform_affine(self, point):
        pass

    def transform_non_affine(self, point):
        return self.transform(point)

    def transform(self, point):
        #Ensure point is a 2D array
        point = _np.atleast_2d(point)
        return _np.concatenate((self.real_interp(point[:, 0], point[:, 1]), self.imag_interp(point[:, 0], point[:, 1])),
                               1)

    def transform_point(self, point):
        t = self.transform(point)
        return t

    def transform_bbox(self, bbox):
        return Bbox.from_extents(self.transform(bbox.get_points()))

    def transform_array(self, data, mode='constant', cval=_np.nan, order=1, prefilter=False):
        # Set output shape
        output_shape = self.lut.shape + data.shape[2:] if data.ndim > 2 else self.lut.shape
        data_gc = _np.zeros(output_shape, dtype=data.dtype).view(type(data))
        x, y = np.arange(data.shape[0]), np.arange(data.shape[1])
        if data.ndim > 2:
            axis_shapes = [list(range(data.shape[i])) for i in range(2, data.ndim)]
            # All combination of axes have to be interpolated on the same 2D grid,
            # therefore we use itertools product function
            for i, axes in enumerate(_iter.product(*axis_shapes)):
                data_gc[(Ellipsis,) * 2 + axes] = interpolate_ndim(x,y,self.lut.real, self.lut.imag, data[(Ellipsis,) * 2 + axes], mode=mode, cval=cval, prefilter=prefilter)
        else:
            data_gc = interpolate_ndim(x,y,self.lut.real, self.lut.imag, data, mode=mode, cval=cval, prefilter=prefilter)
        return data_gc



class GeoTransform(Affine2D):
    """
    Transform DEM-indices to radar indices
    """

    def __init__(self, gt):
        super().__init__(matrix=[[gt[1], gt[2], gt[0]], [gt[4], gt[5], gt[3]], [0, 0, 1]])

# class LutAxes(Axes):
#
#     # name = 'lut'
#     def __init__(self,*args, **kwargs):
#         print(args)
#         _lut = kwargs.pop('lut')
#         _gt = kwargs.pop('gt')
#         Axes.__init__(self, *args)
#         self._lut = ComplexLut(_lut).inverted()
#         self._lut = 1
#         self._gt =3
#
#     def imshow(self,*args, **kwargs):
#
#     def cla(self):
#         Axes.cla(self)
#         min_pt = self.transAffine.transform_point([0,0])
#         max_pt = self.transAffine.transform_point([self._lut.shape[0],self._lut.shape[1]])
#         Axes.set_xlim(self, min_pt[0], max_pt[0])
#         Axes.set_ylim(self, min_pt[1], max_pt[1])
#
#     def _get_non_affine(self):
#         return self._lut
#
#     def _get_affine(self):
#         return GeoTransform(self._gt)
#
#     def _set_lim_and_transforms(self):
#         # 1) Coordinate projection using lookup table
#         self.transProjection = self._get_non_affine()
#
#         # 2) Affine transformation
#         self.transAffine = self._get_affine()
#
#         # 3) Bbox transformation
#         self.transAxes = BboxTransformTo(self.bbox)
#
#         # Data transformation is the composition of
#         # all the previous transforms
#         self.transData = \
#             self.transProjection + \
#             self.transAffine + \
#             self.transAxes
#
#
# class LutProjection():
#
#     def __init__(self,**kwargs):
#         self._gt = kwargs.get('gt')
#         self._lut = kwargs.get('lut')
#
#     def _as_mpl_axes(self):
#         return LutAxes, {'gt':self._gt, 'lut':self._lut}

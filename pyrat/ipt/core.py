import csv

import numpy as _np

from ..fileutils import parameters as par
from ..fileutils.gpri_files import type_mapping as tm


class Plist(object):
    """
    Class to represent a point target list
    """

    def __init__(self, plist_path, r_slc_par_path, **kwargs):
        plist = _np.fromfile(plist_path, dtype=_np.dtype('>i4'))
        self.plist = list(plist.reshape((len(plist)//2, 2)))
        self.params = par.ParameterFile(r_slc_par_path)

    def __getitem__(self, item):
        # Deletage getitem to the numpy array
        return self.plist[item]

    def __getslice__(self, sl):
        return self.plist[sl][:]

    def __iter__(self):
        return self.plist.__iter__()

    def __len__(self):
        return self.plist.__len__()

    def closest_index(self, pos):
        """
        Find the point target index closest to
        the specified coordinate
        Parameters
        ----------
        pos

        Returns
        -------
        """
        plist_arr = _np.array(self.plist)
        residual = _np.sum((plist_arr- _np.array(pos)[None, :]) ** 2, axis=1)
        idx = _np.argmin(residual)
        return idx

    def radar_coords(self, pos):
        # Return the radar coordinates of the point of interest
        r = self.params.near_range_slc + pos[0] * self.params.range_pixel_spacing
        az = self.params.GPRI_az_start_angle + pos[1] * self.params.GPRI_az_angle_step
        return (r, az)

    def cartesian_coord(self, pos):
        (r, az) = self.radar_coords(pos)
        x = r * _np.cos(_np.deg2rad(az))
        y = r * _np.sin(_np.deg2rad(az))
        return (x, y)

    def to_location_list(self):
        res = []
        for idx_pt, pt in enumerate(self):
            coord = self.cartesian_coord(pt)
            res.append(coord)
        return res


class Pdata(object):
    def __init__(self, plist_path, r_slc_par_path, pdata_path, dtype='FCOMPLEX'):
        self.plist = Plist(plist_path, r_slc_par_path)
        pdata = _np.fromfile(pdata_path, dtype=tm[dtype])
        nrecords = len(pdata) // len(self.plist)
        pdata_shape = (nrecords, len(self.plist))
        pdata = pdata.reshape(pdata_shape)
        self.pdata = pdata

    def __getitem__(self, item):
        return self.pdata.__getitem__(item)

    def __getslice__(self, sl):
        return self.pdata.__getslice__(sl)

    @property
    def nrecords(self):
        return self.pdata.shape[0]

    @property
    def ridx(self):
        return [x[0] for x in self.plist]

    @property
    def azidx(self):
        return [x[1] for x in self.plist]

    def to_location_list(self):
        res = []
        for idx_pt, pt in enumerate(self.plist):
            val = self.pdata[:, idx_pt]
            coord = self.plist.cartesian_coord(pt)
            radar_coord = self.plist.radar_coords(pt)
            res.append((list(radar_coord) + list(coord) + list(val)))
        return res

    def to_csv(self, of, take=None):
        dtypes = [('ridx', int), ('azidx', int), ('x', float), ('y', float)] + [
            ('record_{n}'.format(n=n), self.pdata.dtype) for n in range(self.nrecords)]
        headers = [dt[0] for dt in dtypes]
        res = self.to_location_list()[slice(None,None,take)]
        with open(of, 'w+') as out:
            writer = csv.writer(out, delimiter=',')
            writer.writerow(headers)
            for result in res:
                writer.writerow(result)

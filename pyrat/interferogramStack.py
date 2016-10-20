from .fileutils import gpri_files as gpf
import datetime as _dt

dt_string = "%Y %m %d "


class Interferogram(gpf.gammaDataset):
    """
    Class to represent a complex interferogram
    """

    def __new__(cls, *args, **kwargs):
        if "master_par" in kwargs:
            master_par_path = kwargs.pop('master_par')
            slave_par_path = kwargs.pop('slave_par')
            master_par = gpf.par_to_dict(master_par_path)
            slave_par = gpf.par_to_dict(slave_par_path)
            ifgram, ifgram_par = gpf.load_dataset(args[0], args[1], **kwargs)
            ifgram = gpf.gammaDataset(ifgram_par, ifgram)
            ifgram = ifgram.view(cls)
            # geometrical properties that are inherited from GPRI SLC/MLI
            GPRI_prop = ["near_range_slc", "GPRI_az_start_angle", "GPRI_az_angle_step", "range_pixel_spacing", "azimuth_line_time",
                          "prf", "GPRI_ref_north",  "GPRI_ref_east", "GPRI_ref_alt", "GPRI_geoid"]
            for prop in GPRI_prop:
                ifgram.__dict__[prop] = master_par[prop]
            ifgram.__dict__['slave_time'] = gpf.datetime_from_par_dict(master_par)
            ifgram.__dict__['master_time'] = gpf.datetime_from_par_dict(master_par)
        return ifgram

    def to_file(self, par_path, bin_path):
        self = self.astype(gpf.type_mapping['FCOMPLEX'])
        gpf.write_dataset(self, self.__dict__, par_path, bin_path)


class Stack:
    """
    Class to represent a stack of gammaDataset interferograms
    """

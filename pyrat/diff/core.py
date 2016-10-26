from ..fileutils import gpri_files as gpf
import datetime as _dt

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
            ifgram = ifgram.view(cls)
            ifgram._params = ifgram_par.copy()
            #Add properties of master and slave to the dict by adding them and appending
            # for (prop, prop_value) in master_par.items_with_unit():
            #     new_key = 'master_' + prop
            #     print(prop_value)
            #     ifgram.add_parameter(new_key, prop_value['value'], unit=prop_value.get('unit'))
            # # geometrical properties that are inherited from GPRI SLC/MLI
            # GPRI_prop = ["near_range_slc", "GPRI_az_start_angle", "GPRI_az_angle_step", "range_pixel_spacing", "azimuth_line_time",
            #               "prf", "GPRI_ref_north",  "GPRI_ref_east", "GPRI_ref_alt", "GPRI_geoid"]
            # for prop in GPRI_prop:
            #     ifgram.__dict__[prop] = master_par[prop]
            # ifgram.__dict__['slave_time'] = gpf.datetime_from_par_dict(master_par)
            # ifgram.__dict__['master_time'] = gpf.datetime_from_par_dict(master_par)
        return ifgram

    def tofile(self, par_path, bin_path):
        arr = self.astype(gpf.type_mapping['FCOMPLEX'])
        gpf.write_dataset(arr, self._params, par_path, bin_path)
from ..fileutils import gpri_files as gpf
import datetime as _dt

class Interferogram(gpf.gammaDataset):
    """
    Class to represent a complex interferogram
    """

    def __new__(cls, *args, **kwargs):
        ifgram, ifgram_par = gpf.load_dataset(args[0], args[1], **kwargs)
        ifgram = ifgram.view(cls)
        ifgram._params = ifgram_par.copy()
        #Add properties of master and slave to the dict by adding them and appending
        if "master_par" in kwargs:
            master_par_path = kwargs.pop('master_par')
            slave_par_path = kwargs.pop('slave_par')
            master_par = gpf.par_to_dict(master_par_path)
            slave_par = gpf.par_to_dict(slave_par_path)
            for (prop, prop_value) in master_par.items_with_unit():
                new_key = 'master_' + prop
                ifgram.add_parameter(new_key, prop_value['value'], unit=prop_value.get('unit'))
            for (prop, prop_value) in slave_par.items_with_unit():
                new_key = 'slave_' + prop
                ifgram.add_parameter(new_key, prop_value['value'], unit=prop_value.get('unit'))
        return ifgram

    @property
    def temporal_baseline(self):
        return self.master_start_time - self.slave_start_time

    def tofile(self, par_path, bin_path):
        arr = self.astype(gpf.type_mapping['FCOMPLEX'])
        gpf.write_dataset(arr, self._params, par_path, bin_path)
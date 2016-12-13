from ..fileutils import gpri_files as gpf
import datetime as _dt

class Interferogram(gpf.gammaDataset):
    """
    Class to represent a complex interferogram
    """

    def __new__(cls, *args, **kwargs):
        print(args)
        ifgram, ifgram_par = gpf.load_dataset(args[0], args[1], **kwargs)
        ifgram = ifgram.view(cls)
        ifgram._params = ifgram_par.copy()
        #Add properties of master and slave to the dict by adding them and appending
        if "master_par" in kwargs:
            master_par_path = kwargs.pop('master_par')
            slave_par_path = kwargs.pop('slave_par')
            ifgram.master_par = gpf.par_to_dict(master_par_path)
            ifgram.slave_par = gpf.par_to_dict(slave_par_path)
            ifgram.add_parameter('slave_mli', slave_par_path)
            ifgram.add_parameter('master_mli', master_par_path)
        elif "slave_mli" in ifgram._params:
            ifgram.slave_par = gpf.par_to_dict(ifgram._params.slave_mli)
            ifgram.master_par = gpf.par_to_dict(ifgram._params.master_mli)
        return ifgram

    @property
    def temporal_baseline(self):
        return self.master_par.start_time - self.slave_par.start_time

    def tofile(self, par_path, bin_path):
        arr = self.astype(gpf.type_mapping['FCOMPLEX'])
        gpf.write_dataset(arr, self._params, par_path, bin_path)
from ..fileutils import gpri_files as gpf
import datetime as _dt

from . import intfun

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
            ifgram.master_par = gpf.par_to_dict(master_par_path)
            ifgram.slave_par = gpf.par_to_dict(slave_par_path)
            ifgram.add_parameter('slave_par', slave_par_path)
            ifgram.add_parameter('master_par', master_par_path)
        elif "slave_par" in ifgram._params:
            ifgram.slave_par = gpf.par_to_dict(ifgram._params.slave_par)
            ifgram.master_par = gpf.par_to_dict(ifgram._params.master_par)
        return ifgram

    @property
    def temporal_baseline(self):
        return self.master_par.start_time - self.slave_par.start_time

    def tofile(self, par_path, bin_path):
        arr = self.astype(gpf.type_mapping['FCOMPLEX'])
        gpf.write_dataset(arr, self._params, par_path, bin_path)



class Stack:
    """
    Class to represent a stack of interferograms
    """
    def __init__(self, par_list, bin_list, itab, *args, **kwargs):
        stack = []
        for par_name, bin_name in zip(par_list,bin_list):
            ifgram = Interferogram(par_name, bin_name, **kwargs)
            stack.append(ifgram)
        #Sort by acquisition time
        sorting_key = lambda x: (x.master_par.start_time, x.slave_par.start_time)
        stack = sorted(stack, key=sorting_key )
        self.stack  = stack#the stack
        self.itab = intfun.Itab.fromfile(itab)#the corresponding itab file

    @property
    def dt(self):
        return [s.temporal_baseline for s in self.stack]

    def __getitem__(self, item):
        return self.stack.__getitem__(item)

import datetime as _dt

from .parsers import flatify
from . import  parsers as _parsers

import collections as _coll

def format_multiple(par):
    """
    This either formats a string,
    a  list or a single element
    Parameters
    ----------
    par

    Returns
    -------

    """
    if isinstance(par, str):
        par_str = par
    else:
        try:
            par_str = ' '.join(str(x) for x in par)
        except TypeError:
            par_str = str(par)
    return par_str


class ParameterFile(object):
    """
    Class to represent gamma keyword:paramerter files
    """

    @classmethod
    def from_dict(cls, dict):
        instance = ParameterFile()
        instance.params = dict
        instance.add_parameter('file_title',  dict.get('file_title'))
        return instance

    @classmethod
    def from_file(cls, path, parser=_parsers.FasterParser):
        instance = ParameterFile()
        parser = parser()
        with open(path, 'r') as par_text:
            res_dict = parser.as_ordered_dict(par_text.read())
            file_title = res_dict.pop('file_title')
        params = _coll.OrderedDict()
        for key, item in res_dict.items():
            params[key] = {}
            try:
                for (subkey, subitem) in item.items():
                    params[key][subkey] = flatify(subitem)
            except AttributeError:
                params[key]['value'] = flatify(item)
        instance.params = params
        instance.add_parameter('file_title',  file_title)
        return instance



    def __getattr__(self, key):
        try:
            value = self.params[key]['value']
            return value
        except KeyError:
            attr_msg = "The attribute {key} does not exist in the specified parameterfile".format(key=key)
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if 'params' in self.__dict__:
            if key in self.__dict__['params']:
                self.__dict__['params'][key]['value'] = value
            else:
                raise AttributeError("The key: {key} does not exist in the parameterfile, you must first add it with add_parameter".format(key=key, value=value))
        else:
            super(ParameterFile, self).__setattr__(key, value)

    def __getitem__(self, key):
        try:
            return self.params[key]['value']
        except KeyError:
            key_msg = "The attribute {key} does not exist in the specified parameterfile".format(key=key)
            raise KeyError(key_msg)

    def items_with_unit(self):
        return [(key, value) for key, value in self.params.items()]

    def keys(self):
        return self.params.keys()

    def pop(self, key):
        if key in self:
            return self.params.pop(key)['value']

    def items(self):
        return [(key, value['value']) for key,value in self.params.items()]

    def __contains__(self, item):
        if item in self.keys():
            return True
        else:
            return False


    def get(self,key, default=None):
        try:
            return self.params[key]
        except KeyError:
            return default

    def add_parameter(self,key,value, unit=None):
        if key not in self:
            self.params.update({key: {'value': value, 'unit': unit}})

    def copy(self):
        new_params = {}
        for key, value in self.params.items():
            new_params[key] = value.copy()
        # params = _cp.deepcopy(self.params.copy())
        new_pf = ParameterFile.from_dict(new_params)
        return new_pf


    def __setitem__(self, key, value, unit=None):
        if 'params' in self.__dict__:
            try:
                self.params[key]['value'] = value
            except KeyError:
                raise KeyError(
                    "{key} does not exist in the specified parameterfile, use  add_parameter to add it to the file".format(
                        key=key))

    def format_key_unit_dict(self, key):
        """
        Used to format a dict of form
        {value: single value or list of values, unit: nothing or list of units}
        Parameters
        ----------
        dict

        Returns
        -------

        """
        value = self.params[key]['value']
        unit = self.params[key].get('unit')
        value_str = format_multiple(value)
        if unit:
            unit_str = format_multiple(unit)
        else:
            unit_str = ''
        return value_str + ' ' + unit_str



    def plain_dict(self):
        """
        Returns a plain dict, where only the numerical values are stored

        Returns
        -------

        """
        return {key: item['value'] for (key, item) in dict(ParameterFile).items()}

    def unit_mapping(self):
        """
        Returns a plain dict, where only the numerical values are stored

        Returns
        -------

        """
        return {key: item['unit'] for (key, item) in iter(self)}


    def __str__(self):
        self_1 = self.copy()
        out_str = ""
        if 'file_title' in self_1:
            title = self_1.params.pop('file_title')
            if title['value'] is not None:
                out_str += title['value'] + '\n'
        for key, value in self_1.params.items():
            par_str = self_1.format_key_unit_dict(key)
            key_str = "{key}:".format(key=key).ljust(40)
            par_str_just = par_str.rjust(20)
            line = "{key} {par_str}\n".format(key=key_str, par_str=par_str_just)
            out_str += line
        return out_str

    def tofile(self, par_file):
        if isinstance(par_file, str):
            with open(par_file, 'w+') as fout:
                fout.writelines(str(self))
        elif hasattr(par_file, 'read'):
            par_file.writelines(str(self))


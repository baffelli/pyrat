import datetime as _dt

from fileutils.parsers import flatify


def dt_parse(s, l, t):
    if 'time' in t[0]:
        tim = _dt.datetime(t[0]['year'], t[0]['month'], t[0]['day'],t[0]['time']['h'],t[0]['time']['m'],t[0]['time']['s'],t[0]['time']['us'])
    else:
        tim = _dt.date(t[0]['year'], t[0]['month'], t[0]['day'])
    dt_dict = {'value':
                   tim, 'unit': None}
    return dt_dict


def int_parse(s, l, t):
    return int(t[0])


def multiline_parse(s, l, t):
    """
    Parsing action for multiline text,
    returns a single string joined by newlines
    Parameters
    ----------
    s
    l
    t

    Returns
    -------

    """
    return "\n".join(t)


def ambigous_int_parse(s, l, t):
    """
    This is a parse action for

    the sort of ints that were saved by the previous dict_to_par as 1.0
    Parameters
    ----------
    s
    l
    t

    Returns
    -------

    """
    try:
        parsed = int(t[0])
    except ValueError:
        parsed = int(t[0].split('.')[0])
    return parsed


def float_parse(s, l, t):
    return float(t[0])


def text_parse(s, l, t):
    t1 = t.asDict()
    t1['value'] = t1['value'].strip()
    return t1






class arrayContainer:
    """
    Class to collect recusrive parsing output inside  of a dict
    """
    ""

    def __init__(self):
        self.dict = {}

    def array_parse(self, s, l, t):
        for key in t.keys():
            if key in self.dict:#Units and keys are parsed from left to right and vice versa, hence they need to be inserted in a different order
                if key == 'unit':
                    self.dict[key].append(t[key])
                elif key == 'value':
                    self.dict[key].insert(0, t[key])
            else:
                self.dict[key] = [t[key]]



        return self.dict

    def reset(self):
        self.dict = {}
        self.count = 0


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
    def from_file(cls, path):
        instance = ParameterFile()
        parser = ()
        with open(path, 'r') as par_text:
            res_dict = parser.as_ordered_dict(par_text.read())
            file_title = res_dict.pop('file_title')
        params = {}
        for key, item in res_dict.items():
            params[key] = {}
            for (subkey, subitem) in item.items():
                params[key][subkey] = flatify(subitem)
        instance.params = params
        instance.add_parameter('file_title',  file_title)
        return instance



    # def __init__(self, *args):
    #     if isinstance(args[0], str):
    #         parser = ParameterParser()
    #         with open(args[0], 'r') as par_text:
    #             # parsedResults = parser.parse(par_text.read())
    #             res_dict = parser.as_ordered_dict(par_text.read())
    #             file_title = res_dict.pop('file_title')
    #     elif hasattr(args[0], 'items'):
    #         res_dict = args[0]
    #         file_title = res_dict.get('file_title')
    #     params = _coll.OrderedDict()
    #     params = {}
    #     for key, item in res_dict.items():
    #         params[key] = {}
    #         for (subkey, subitem) in item.items():
    #             params[key][subkey] = flatify(subitem)
    #     # params['file_title'] = {'value': file_title, 'unit': None}
    #     # self.file_title = file_title
    #     self.params = params
    #     self.add_parameter('file_title', file_title)

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

    # def __getattr__(self, key):
    #     dict_item = super(ParameterFile,self).__getitem__(key)
    #     try:
    #         item = dict_item['value']
    #         try:#single element list
    #             singleitem, = item
    #         except (TypeError, ValueError):#complete list
    #             return item
    #         else:
    #             return singleitem
    #     except KeyError:
    #         return dict_item

    # def __setattr__(self, key, value):
    #     try:
    #         super(ParameterFile,self).__setitem__(key,{'value': value})
    #     except KeyError:
    #         pass
    #
    # def __getitem__(self, item):
    #     dict_item = super(ParameterFile, self).__getitem__(item)
    #     try:
    #         item = dict_item['value']
    #         try:#single element list
    #             singleitem, = item
    #         except (TypeError, ValueError):#complete list
    #             return item
    #         else:
    #             return singleitem
    #     except KeyError:
    #         return dict_item

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


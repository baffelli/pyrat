import collections as _coll
import datetime as _dt
import copy as _cp
import pyparsing as _pp


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



def compound_unit_parse(s, l, t):
    """
    Join the elements of a compound unit
    Parameters
    ----------
    s
    l
    t

    Returns
    -------

    """
    if len(t['unit']) > 1:
        return ''.join(t['unit'])
    elif len(t['unit']) == 1:
        return t['unit'][0]


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


def flatify(arr):
    """
    Flatten a one-elmenent array
    Parameters
    ----------
    arr

    Returns
    -------

    """
    try:
        item, = arr
        return item
    except (TypeError, ValueError):
        return arr


class ParameterParser:
    def __init__(self):
        self.array_container = arrayContainer()
        EOL = _pp.LineEnd().suppress().setParseAction(self.array_container.reset)
        SOL = _pp.LineStart().suppress()
        _pp.ParserElement.setDefaultWhitespaceChars(' \t')
        self.grammar = type('grammar', (object,), {})()
        # separator of keyword
        self.grammar.keyword_sep = _pp.Literal(':').suppress()
        # Optional date seprator
        self.grammar.date_sep = _pp.Literal('-').suppress()
        # any text
        self.grammar.text_parameter = _pp.Combine(_pp.restOfLine())
        # Definition of unit
        self.grammar.base_units = _pp.Literal('dB') | _pp.Literal('s') | _pp.Literal('m') | _pp.Literal(
            'Hz') | _pp.Literal('degrees') | _pp.Literal('arc-sec') | _pp.Literal('decimal degrees') | _pp.Literal('1')
        self.grammar.repeated_unit = (self.grammar.base_units + _pp.Optional(_pp.Literal('^') + _pp.Optional(_pp.Literal('-')) +_pp.Word(_pp.nums)))
        self.grammar.unit = _pp.Group(
            self.grammar.repeated_unit + _pp.Optional(_pp.ZeroOrMore('/' + self.grammar.repeated_unit)))(
            'unit').setParseAction(compound_unit_parse)
        # definition of numbers
        self.grammar.float_re = _pp.Regex('[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?').setParseAction(float_parse)
        self.grammar.int = _pp.Word(_pp.nums).setParseAction(int_parse)
        self.grammar.number = (self.grammar.int ^ self.grammar.float_re)('value')
        # Recursive Definition of array:
        array = _pp.Forward()
        array << (
            (self.grammar.number + array + _pp.Optional(self.grammar.unit)) | (
                self.grammar.number + _pp.Optional(self.grammar.unit)))
        self.grammar.array = array.setParseAction(self.array_container.array_parse)
        # Date (year-month-day or year month day)
        # Extendend date
        self.grammar.time = _pp.Group(
            _pp.Word(_pp.nums)('h').setParseAction(
                ambigous_int_parse) + _pp.Literal(':') + _pp.Word(_pp.nums)('m').setParseAction(
                ambigous_int_parse) + _pp.Literal(':') + _pp.Word(
                _pp.nums )('s').setParseAction(
                ambigous_int_parse)  + _pp.Literal('.') + _pp.Word(_pp.nums)('us').setParseAction(
                ambigous_int_parse) + _pp.Optional(_pp.Group(_pp.Literal('+') + _pp.Word(_pp.nums)('tz_h').setParseAction(
                ambigous_int_parse) + _pp.Literal(':') + _pp.Word(_pp.nums)('tz_m').setParseAction(
                ambigous_int_parse))))('time')
        self.grammar.date = _pp.Group(
            _pp.Word(_pp.nums + '.',min=4, max=6)('year').setParseAction(
                ambigous_int_parse) + _pp.Optional(
                self.grammar.date_sep) + _pp.Word(
                _pp.nums + '.',min=2, max=4)(
                'month').setParseAction(ambigous_int_parse) + _pp.Optional(self.grammar.date_sep) + _pp.Word(
                _pp.nums + '.', min=2, max=4)(
                'day').setParseAction(
                ambigous_int_parse) + _pp.Optional(self.grammar.time) ).setParseAction(dt_parse)
        self.grammar.title = _pp.Group(
            _pp.Literal('title') + self.grammar.keyword_sep + self.grammar.text_parameter('value').setParseAction(
                text_parse))
        # normal keyword
        self.grammar.normal_kw = _pp.Word(
            _pp.alphanums + '_') + self.grammar.keyword_sep
        # line of normal values
        self.grammar.normal_line = _pp.Group(self.grammar.normal_kw + ( self.grammar.date ^
            self.grammar.array ^
            self.grammar.text_parameter('value').setParseAction(text_parse)))
        # title
        self.grammar.file_title = _pp.Group(_pp.Combine(_pp.ZeroOrMore((SOL + ~(self.grammar.normal_kw) + self.grammar.text_parameter + _pp.LineEnd())))(
            'value'))('file_title')
        # Normal line
        self.grammar.line = (_pp.Dict(
            (self.grammar.normal_line) | _pp.Dict(self.grammar.title))) + EOL
        self.grammar.param_grammar = _pp.Optional(self.grammar.file_title) + (_pp.OneOrMore(
            self.grammar.line) | _pp.ZeroOrMore(EOL))

    def parse(self, text_object):
        return self.grammar.param_grammar.parseString(text_object)

    def as_ordered_dict(self, text_object):
        parsed = self.parse(text_object)
        result_dict = _coll.OrderedDict()
        for p in parsed.asList():#really ugly way to obtaine ordered results, until "asDict()" supports
            try:
                name, value = p
                result_dict[name] = value
            except ValueError:
                result_dict['file_title'] = flatify(p)
        return result_dict


class ParameterFile(object):
    """
    Class to represent gamma keyword:paramerter files
    """

    def __init__(self, *args):
        if isinstance(args[0], str):
            parser = ParameterParser()
            with open(args[0], 'r') as par_text:
                # parsedResults = parser.parse(par_text.read())
                res_dict = parser.as_ordered_dict(par_text.read())
                file_title = res_dict.pop('file_title')
        elif hasattr(args[0], 'items'):
            res_dict = args[0]
            file_title = res_dict.get('file_title')
        params = _coll.OrderedDict()
        for key, item in res_dict.items():
            params[key] = _coll.OrderedDict()
            for (subkey, subitem) in item.items():
                params[key][subkey] = flatify(subitem)
        # params['file_title'] = {'value': file_title, 'unit': None}
        # self.file_title = file_title
        self.params = params
        self.add_parameter('file_title', file_title)

    def __getattr__(self, key):
        try:
            return self.params[key]['value']
        except KeyError:
            attr_msg = "The attribute {key} does not exist in the specified parameterfile".format(key=key)
            raise AttributeError(attr_msg)

    def __setattr__(self, key, value):
        if 'params' in self.__dict__:
            if key in self.__dict__['params']:
                self.__dict__['params'][key]['value'] = value
            else:
                raise AttributeError("{key} does not exist in the specified parameterfile, use  add_parameter to add it to the file".format(key=key))
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


    def get(self,key):
        try:
            return self.params[key]
        except KeyError:
            return None


    def add_parameter(self,key,value, unit=None):
        if key not in self:
            self.params.update({key: {'value': value, 'unit': unit}})

    def copy(self):
        params = _cp.deepcopy(self.params)
        new_pf = ParameterFile(params)
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


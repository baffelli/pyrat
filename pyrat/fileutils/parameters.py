import collections as _coll
import datetime as _dt

import pyparsing as _pp


def dt_parse(s, l, t):
    dt_dict = {'value':
    _dt.date(t[0]['year'], t[0]['month'], t[0]['day']), 'unit': None}
    return dt_dict


def int_parse(s, l, t):
    return int(t[0])


def multiline_parse(s,l,t):
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
    return t.asDict()


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

    def array_parse(self, s,l,t):
        for key in t.keys():
            if key in self.dict:
                self.dict[key].append((t[key]))
            else:
                self.dict[key] = [t[key]]
        return self.dict

    def reset(self):
        self.dict = {}


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
    elif hasattr(par, '__getitem__'):
        par_str = ' '.join(str(x) for x in par)
    else:
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


class parameterParser:
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
        #any text
        self.grammar.text_parameter = _pp.Combine(_pp.restOfLine())
        # Definition of unit
        self.grammar.base_units = _pp.Literal('dB') | _pp.Literal('s') | _pp.Literal('m') | _pp.Literal(
            'Hz') | _pp.Literal('degrees') | _pp.Literal('arc-sec') | _pp.Literal('decimal degrees') | _pp.Literal('1')
        self.grammar.repeated_unit = (self.grammar.base_units + _pp.Optional('^' + _pp.Word('-123')))
        self.grammar.unit = _pp.Group(
            self.grammar.repeated_unit + _pp.Optional(_pp.ZeroOrMore('/' + self.grammar.repeated_unit)))(
            'unit').setParseAction(compound_unit_parse)
        #definition of numbers
        self.grammar.float_re = _pp.Regex('[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?').setParseAction(float_parse)
        self.grammar.int = _pp.Word(_pp.nums).setParseAction(int_parse)
        self.grammar.number = (self.grammar.float_re | self.grammar.int)('value')
        # Recursive Definition of array:
        array = _pp.Forward()
        array << (
            (self.grammar.number + array + _pp.Optional(self.grammar.unit)) | (
                self.grammar.number + _pp.Optional(self.grammar.unit)))
        self.grammar.array = array.setParseAction(self.array_container.array_parse)
        # Date (year-month-day or year month day)
        self.grammar.date = _pp.Group(_pp.Literal('date') + self.grammar.keyword_sep + _pp.Group(
            _pp.Word(_pp.nums + '.')('year').setParseAction(
                ambigous_int_parse) + _pp.Optional(
                self.grammar.date_sep) + _pp.Word(
                _pp.nums + '.')(
                'month').setParseAction(ambigous_int_parse) + _pp.Optional(self.grammar.date_sep) + _pp.Word(
                _pp.nums + '.')(
                'day').setParseAction(
                ambigous_int_parse)).setParseAction(dt_parse))
        self.grammar.title = _pp.Group(_pp.Literal('title') + self.grammar.keyword_sep + self.grammar.text_parameter('value').setParseAction(text_parse))
        # normal keyword
        self.grammar.normal_kw = ~_pp.Literal('date') + ~_pp.Literal('title') + _pp.Word(_pp.alphanums + '_') + self.grammar.keyword_sep
        # line of normal values
        self.grammar.normal_line = _pp.Group(self.grammar.normal_kw + (
            self.grammar.array |
            self.grammar.text_parameter('value').setParseAction(text_parse)))
        # title
        self.grammar.file_title = _pp.Combine(_pp.ZeroOrMore((SOL + ~(_pp.Group(_pp.Word(_pp.alphanums + '_') + self.grammar.keyword_sep)) + self.grammar.text_parameter + _pp.LineEnd())))('file_title')
        # Normal line
        self.grammar.line = (_pp.Dict((self.grammar.normal_line) | _pp.Dict(self.grammar.date) | _pp.Dict(self.grammar.title))) + EOL
        self.grammar.param_grammar = _pp.Optional(self.grammar.file_title) + (_pp.OneOrMore(
            self.grammar.line) & _pp.ZeroOrMore(EOL))

    def parse(self, text_object):
        return self.grammar.param_grammar.parseString(text_object)

    def as_ordered_dict(self, text_object):
        parsed = self.parse(text_object)
        result_dict = _coll.OrderedDict(parsed.asDict())
        return result_dict

class ParameterFile(object):
    """
    Class to represent gamma keyword:paramerter files
    """

    def __init__(self, *args):
        if isinstance(args[0], str):
            parser = parameterParser()
            with open(args[0], 'r') as par_text:
                # parsedResults = parser.parse(par_text.read())
                res_dict = parser.as_ordered_dict(par_text.read())
        elif hasattr('get', args[0]):
            res_dict = args[0]
        file_title = res_dict.pop('file_title')
        params = _coll.OrderedDict()
        for key, item in res_dict.items():
            params[key] = _coll.OrderedDict()
            for (subkey, subitem) in item.items():
                params[key][subkey] = flatify(subitem)
        self.file_title = file_title
        self.params = params




    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            try:
                return self.params[key]['value']
            except KeyError:
                raise AttributeError("This attribute does not exist in the specified parameterfile")




    def __setattr__(self, key, value):
        if 'params' in self.__dict__:
            if key in self.__dict__['params']:
                self.__dict__['params'][key]['value'] = value
        else:
            super(ParameterFile, self).__setattr__(key, value)


    def __getitem__(self, key):
        return self.params[key]['value']

    def __setitem__(self, key, value):
        self.params[key]['value'] = value

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

    def to_file(self, par_file):
        with open(par_file, 'w') as fout:
            if hasattr(self, 'file_title'):
                fout.write(self.file_title)
            for key, value in self.params.items():
                par_str= self.format_key_unit_dict(key)
                key_str = "{key}:".format(key=key).ljust(40)
                par_str_just = par_str.ljust(20)
                line = "{key} {par_str}\n".format(key=key_str, par_str=par_str_just)
                fout.write(line)

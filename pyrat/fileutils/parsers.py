import collections as _coll
import datetime as _dt

import pyparsing as _pp

# from fileutils.parameters import arrayContainer, compound_unit_parse, float_parse, int_parse, ambigous_int_parse, \
#     dt_parse, text_parse

_pp.ParserElement.setDefaultWhitespaceChars(' \t')

#Shared parse actions

def int_parse(s, l, t):
    return int(t[0])


def float_parse(s, l, t):
    return float(t[0])


def text_parse(s, l, t):
    t1 = t.asDict()
    t1['value'] = t1['value'].strip()
    t1['unit'] = None
    return t1

def strip_text(s,l,t):
    return t[0].strip()


def dt_parse(s, l, t):
    tinfo = t.asDict()['datetime']
    if 'time' in tinfo:
        tim = _dt.datetime(tinfo['date']['year'], tinfo['date']['month'],
                           tinfo['date']['day'], tinfo['time']['hour'],
                           tinfo['time']['minute'], second=tinfo['time']['second'],
                           microsecond=tinfo['time']['ms'])
    else:
        tim = _dt.date(tinfo['date']['year'], tinfo['date']['month'],
                       tinfo['date']['day'])
    return tim


def dt_parse_old(s, l, t):
    if 'time' in t[0]:
        tim = _dt.datetime(t[0]['year'], t[0]['month'], t[0]['day'],t[0]['time']['h'],t[0]['time']['m'],t[0]['time']['s'],t[0]['time']['us'])
    else:
        tim = _dt.date(t[0]['year'], t[0]['month'], t[0]['day'])
    dt_dict = {'value':
                   tim, 'unit': None}
    return dt_dict

def strip_white(s,l,t):
    return t[0].strip()

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

def unit_parse(s,l,t):
    if len(t) == 0:
        return []
    else:
        return t

def array_parse(s,l,t):
    if len(t) == 0:
        return t[0]
    else:
        return t


class arrayContainer:
    """
    Class to collect recusrive parsing output inside  of a dict
    """
    ""

    def __init__(self):
        self.dict = {}

    def array_parse(self, s, l, t):
        for key in t.keys():
            if key in self.dict:  # Units and keys are parsed from left to right and vice versa, hence they need to be inserted in a different order
                if key == 'unit':
                    self.dict[key].append(t[key])
                elif key == 'value':
                    self.dict[key].insert(0, t[key])
            else:
                self.dict[key] = [t[key]]
        return self.dict

    def reset(self):
        self.dict = {}


class FasterParser:
    def __init__(self):
        _pp.ParserElement.setDefaultWhitespaceChars(' \t')
        array_container = arrayContainer()
        # End of line
        EOL = _pp.LineEnd().suppress().setParseAction(array_container.reset)
        SOL = _pp.LineStart().suppress()
        # Keyword separator
        KW_SEP = _pp.Literal(':').suppress()
        HMS_SEP = _pp.Literal(':').suppress()
        # Date separator
        DT_SEP = _pp.Literal('-').suppress()
        # Decimal dot
        DDOT = _pp.Literal('.').suppress()
        # Timezone symbol
        TZ_SEP = _pp.Literal('+').suppress()
        # Parameter name
        parameter_name = _pp.Word(_pp.alphanums + '_')
        # Text parameter
        regular_text = _pp.OneOrMore(_pp.Word(_pp.printables).setParseAction(strip_white).setParseAction(strip_text))
        # Date parameter
        year = _pp.Word(_pp.nums + '.', min=4, max=6)('year').setParseAction(int_parse)
        month = _pp.Word(_pp.nums + '.', min=2, max=4)('month').setParseAction(int_parse)
        day = _pp.Word(_pp.nums + '.', min=2, max=4)('day').setParseAction(int_parse)
        hour = _pp.Word(_pp.nums, min=2, max=2)('hour').setParseAction(int_parse)
        minute = _pp.Word(_pp.nums, min=2, max=2)('minute').setParseAction(int_parse)
        second = _pp.Word(_pp.nums, min=2, )('second').setParseAction(int_parse)
        millisecond = _pp.Word(_pp.nums)('ms').setParseAction(int_parse)
        # Date
        date = _pp.Group(year + _pp.Optional(DT_SEP) + month + _pp.Optional(DT_SEP) + day)('date')
        # Hour minute second
        local_time = _pp.Group(hour + HMS_SEP + minute + HMS_SEP + second + DDOT + millisecond)('time')
        # timezone info
        tzinfo = _pp.Group(TZ_SEP + hour + HMS_SEP + minute )('tzinfo')
        datetime = _pp.Group(date + _pp.Optional(local_time + _pp.Optional(tzinfo)))('datetime').setParseAction(dt_parse)
        # Numbers
        float = _pp.Regex('[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?').setParseAction(float_parse)
        int = _pp.Word(_pp.nums).setParseAction(int_parse)
        number = (int ^ float)
        #Array is composed of unit and numbers
        unit = _pp.Word(_pp.alphanums + '/' + '^' + '-')
        array = _pp.Group(_pp.OneOrMore(number))('value').setParseAction(array_parse) + _pp.ZeroOrMore(unit).setParseAction(unit_parse)('unit')
        # Keyword
        kw = parameter_name + KW_SEP
        #Undefined line
        unparsed =  _pp.Combine(_pp.restOfLine()).setParseAction(strip_white)
        #A line is either a datetime object, a regular text or unparsed text
        line_value = (array | regular_text | unparsed | datetime)('value')
        # Line
        normal_kwpair = _pp.Dict(_pp.Group( kw + line_value))
        #The line containing "date" requires a special parsing
        date_kwpair = _pp.Dict((_pp.Group((_pp.Word('date') | _pp.Word('time_start')) + KW_SEP + datetime)))
        #The same applied to the title
        title_kwpair = _pp.Dict((_pp.Group((_pp.Word('title')) + KW_SEP + unparsed)))
        line = _pp.Optional(SOL) + (date_kwpair ^ title_kwpair ^ normal_kwpair)  + _pp.Optional(EOL)
        empty_line = EOL
        # Title
        file_title = _pp.Group(_pp.Combine(_pp.ZeroOrMore(SOL + ~(kw) + unparsed + _pp.LineEnd())))('file_title')
        self.grammar = _pp.Optional(file_title) + (_pp.ZeroOrMore(line)  | _pp.ZeroOrMore(empty_line))

    def parse(self, file):
        parsed = self.grammar.parseString(file)
        return parsed

    def as_ordered_dict(self, text_object):
        parsed = self.parse(text_object)
        result_dict = _coll.OrderedDict()
        for p in parsed.asList():  # really ugly way to obtaine ordered results, until "asDict()" supports
            try:
                name, value, *unit = p
                result_dict[name] = {'value': flatify(value), 'unit':  flatify(unit)}
                # result_dict[name]['unit'] =
            except ValueError:
                result_dict['file_title'] = flatify(p)
        return result_dict




class FastestParser:

    def __init__(self):
        _pp.ParserElement.setDefaultWhitespaceChars(' \t')
        array_container = arrayContainer()
        # End of line
        EOL = _pp.LineEnd().suppress()
        SOL = _pp.LineStart().suppress()
        # Keyword separator
        KW_SEP = _pp.Literal(':').suppress()
        HMS_SEP = _pp.Literal(':').suppress()
        # Date separator
        DT_SEP = _pp.Literal('-').suppress()
        # Decimal dot
        DDOT = _pp.Literal('.').suppress()
        # Timezone symbol
        TZ_SEP = _pp.Literal('+').suppress()
        # Parameter name
        parameter_name = _pp.Word(_pp.alphanums + '_')
        # Text parameter
        regular_text = _pp.OneOrMore(_pp.Word(_pp.printables).setParseAction(strip_white).setParseAction(strip_text))
        # Date parameter
        year = _pp.Word(_pp.nums + '.', min=4, max=6)('year').setParseAction(int_parse)
        month = _pp.Word(_pp.nums + '.', min=2, max=4)('month').setParseAction(int_parse)
        day = _pp.Word(_pp.nums + '.', min=2, max=4)('day').setParseAction(int_parse)
        hour = _pp.Word(_pp.nums, min=2, max=2)('hour').setParseAction(int_parse)
        minute = _pp.Word(_pp.nums, min=2, max=2)('minute').setParseAction(int_parse)
        second = _pp.Word(_pp.nums, min=2, )('second').setParseAction(int_parse)
        millisecond = _pp.Word(_pp.nums)('ms').setParseAction(int_parse)
        # Date
        date = _pp.Group(year + _pp.Optional(DT_SEP) + month + _pp.Optional(DT_SEP) + day)('date')
        # Hour minute second
        local_time = _pp.Group(hour + HMS_SEP + minute + HMS_SEP + second + DDOT + millisecond)('time')
        # timezone info
        tzinfo = _pp.Group(TZ_SEP + hour + HMS_SEP + minute )('tzinfo')
        datetime = _pp.Group(date + _pp.Optional(local_time + _pp.Optional(tzinfo)))('datetime').setParseAction(dt_parse)
        # Numbers
        float = _pp.Regex('[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?').setParseAction(float_parse)
        int = _pp.Word(_pp.nums).setParseAction(int_parse)
        number = (float | int)
        #Array is composed of unit and numbers
        # unit = _pp.Word(_pp.alphanums + '/' + '^' + '-')
        array = _pp.Group(_pp.OneOrMore(number))('value').setParseAction(array_parse) + _pp.restOfLine().setParseAction(unit_parse)('unit')
        # Keyword
        kw = parameter_name + KW_SEP
        #Undefined line
        unparsed =  _pp.Combine(_pp.restOfLine()).setParseAction(strip_white)
        #A line is either a datetime object, a regular text or unparsed text
        line_value = (array | regular_text | unparsed | datetime)('value')
        # Line
        normal_kwpair = _pp.Dict(_pp.Group( kw + line_value))
        #The line containing "date" requires a special parsing
        date_kwpair = _pp.Dict((_pp.Group((_pp.Literal('date') | _pp.Literal('time_start')) + KW_SEP + datetime)))
        #The same applied to the title
        title_kwpair = _pp.Dict((_pp.Group((_pp.Word('title')) + KW_SEP + unparsed)))
        line = (_pp.Optional(SOL) + (date_kwpair ^ title_kwpair ^ normal_kwpair)  + _pp.Optional(EOL)) | EOL
        # Title
        file_title = _pp.Group(_pp.Combine(_pp.ZeroOrMore(SOL + ~(kw) + unparsed )))('file_title')
        self.grammar = _pp.Optional(file_title) + _pp.ZeroOrMore(line)

    def parse(self, file):
        parsed = self.grammar.parseString(file)
        return parsed

    def as_ordered_dict(self, text_object):
        parsed = self.parse(text_object)
        result_dict = _coll.OrderedDict()
        for p in parsed.asList():  # really ugly way to obtaine ordered results, until "asDict()" supports
            try:
                name, value, *unit = p
                result_dict[name] = {'value': flatify(value), 'unit':  flatify(unit)}
                # result_dict[name]['unit'] =
            except ValueError:
                result_dict['file_title'] = flatify(p)
        return result_dict


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
                ambigous_int_parse) + _pp.Optional(self.grammar.time) ).setParseAction(dt_parse_old)
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
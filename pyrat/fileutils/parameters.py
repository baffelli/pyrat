import collections as coll
import datetime as _dt

import pyparsing as _pp


def dt_parse(s, l, t):
    return _dt.date(t[0]['year'], t[0]['month'], t[0]['day'])


def int_parse(s, l, t):
    return int(t[0])


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
    if len(t[0]) == 1:
        return t[0][0]


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


def array_parse(s, l, t):
    print(t.asDict())

    # if len(t[0]) > 1:
    #     return t
    # else:
    #     return t[0][0]

# def title_parser(s,l,t):
#     print('title')
#     print(t['file_title'])
#     return t

class parameterParser:
    def __init__(self):
        EOL = _pp.LineEnd().suppress()
        SOL = _pp.LineStart().suppress()
        _pp.ParserElement.setDefaultWhitespaceChars(' \t')
        self.grammar = type('grammar', (object,), {})()
        # separator of keyword
        self.grammar.keyword_sep = _pp.Literal(':').suppress()
        # Optional date seprator
        self.grammar.date_sep = _pp.Literal('-').suppress()
        #any text
        self.grammar.text_parameter = _pp.Group(_pp.Combine(
            _pp.restOfLine()))
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
        self.grammar.number = (self.grammar.float_re | self.grammar.int)('number')
        # Recursive Definition of array:
        array = _pp.Forward()
        array << (
            (self.grammar.number + array + _pp.Optional(self.grammar.unit)) | (
                self.grammar.number + _pp.Optional(self.grammar.unit)))
        self.grammar.array = array.setParseAction(array_parse)
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
        # normal keyword
        self.grammar.normal_kw = ~_pp.Literal('date') + _pp.Word(_pp.alphanums + '_') + self.grammar.keyword_sep
        # line of normal values
        self.grammar.normal_line = self.grammar.normal_kw + (_pp.Group(
            self.grammar.array) | _pp.Group(
            self.grammar.text_parameter).setParseAction(text_parse))
        # title
        self.grammar.title = ~self.grammar.normal_kw + self.grammar.text_parameter('file_title')
        # Normal line
        self.grammar.line = (_pp.Dict(
            _pp.Group(self.grammar.normal_line)) | _pp.Dict(self.grammar.date)) + EOL
        self.grammar.param_grammar = _pp.Optional(self.grammar.title) & _pp.OneOrMore(
            self.grammar.line) & _pp.ZeroOrMore(EOL)

    def parse(self, text_object):
        return self.grammar.param_grammar.parseString(text_object)


class ParameterFile(coll.OrderedDict):
    """
    Class to represent gamma keyword:paramerter files
    """

    def __init__(self, *args):
        if isinstance(args[0], str):
            parser = parameterParser()
            with open(args[0], 'r') as par_text:
                parsedResults = parser.parse(par_text.read())
            title = parsedResults.pop('file_title')
            print(parsedResults.asDict())
            mapping = [(toks[0], toks[1:][0]) for toks in parsedResults.asList()]
            super(ParameterFile, self).__init__(mapping)
            self.__dict__.update(self)
            self.file_title = 'a'

    def to_file(self, par_file):
        with open(par_file, 'w') as fout:
            if self.file_title:
                fout.write(self.file_title + '\n')
            for key in iter(self):
                par = self[key]
                if isinstance(par, str):
                    par_str = par
                elif hasattr(par, '__getitem__'):
                    par_str = ' '.join(str(x) for x in par)
                else:
                    par_str = str(par)
                key_str = "{key}:".format(key=key).ljust(20)
                par_str_just = par_str.ljust(20)
                line = "{key} {par_str}\n".format(key=key_str, par_str=par_str_just)
                fout.write(line)

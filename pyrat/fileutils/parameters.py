import datetime as _dt

import pyparsing as _pp


def dt_parse(s, l, t):
    return _dt.date(t['value']['year'], t['value']['month'], t['value']['day'])


def int_parse(s, l, t):
    return int(t[0])

def text_parse(s,l,t):
    if len(t[0]) == 1:
        return t[0][0]
    else:
        return t

def unit_parse(s, l, t):
    if len(t['unit']) > 1:
        return ''.join(t['unit'])
    elif len(t['unit']) == 1:
        return t['unit']
    elif t['unit'] == 's':
        return t


def array_parse(s, l, t):
    print(t)
    if len(t[0]) == 1:
        return t[0]
    elif 'unit' in t[0]:
        if t[0]['unit'] == 's':
            return _dt.timedelta(t[0][0])
    else:
        return t


class ParameterFile:
    def __init__(self):
        EOL = _pp.LineEnd().suppress()
        _pp.ParserElement.setDefaultWhitespaceChars(' \t')
        self.grammar = type('grammar', (object,), {})()
        # separator of keyword
        self.grammar.keyword_sep = _pp.Literal(':').suppress()
        self.grammar.text_parameter = _pp.Group(_pp.Combine(
            _pp.restOfLine())).setResultsName('value')
        # Definition of numbers
        self.grammar.base_units = _pp.Literal('dB') | _pp.Literal('s') | _pp.Literal('m') | _pp.Literal(
            'Hz') | _pp.Literal('degrees') | _pp.Literal('1')
        self.grammar.repeated_unit = (self.grammar.base_units + _pp.Optional('^' + _pp.Word('-123')))
        self.grammar.unit = _pp.Group(
            self.grammar.repeated_unit + _pp.Optional(_pp.ZeroOrMore('/' + self.grammar.repeated_unit))).setResultsName(
            'unit').setParseAction(unit_parse)
        self.grammar.float_re = _pp.Regex('[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?').setParseAction(
            lambda s, l, t: float(t[0]))
        self.grammar.int = _pp.Word(_pp.nums).setParseAction(int_parse)
        self.grammar.number = (self.grammar.float_re | self.grammar.int).setResultsName('number')
        # Definition of array:
        array = _pp.Forward()
        array << (
            (self.grammar.number + array + _pp.Optional(self.grammar.unit)) | (
            self.grammar.number + _pp.Optional(self.grammar.unit)))
        self.grammar.array = array
        # Date
        self.grammar.date = _pp.Literal('date') + self.grammar.keyword_sep + _pp.Group(
            _pp.Word(_pp.nums + '.').setResultsName('year').setParseAction(int_parse) + _pp.Word(
                _pp.nums + '.').setResultsName(
                'month').setParseAction(int_parse) + _pp.Word(_pp.nums + '.').setResultsName('day').setParseAction(
                int_parse)).setResultsName('value').setParseAction(dt_parse)
        # Line with general text
        self.grammar.ascii_kw = _pp.Literal('title') | _pp.Literal('sensor') | _pp.Literal('image_format') | _pp.Literal(
                'image_geometry') | _pp.Literal('azimuth_deskew') | _pp.Literal('GPRI_TX_mode') | _pp.Literal(
                'GPRI_TX_antenna')
        # normal keyword
        self.grammar.normal_kw = ~_pp.Literal('date') + ~self.grammar.ascii_kw + _pp.Word(_pp.alphanums + '_')
        self.grammar.normal_line = self.grammar.normal_kw + self.grammar.keyword_sep + _pp.Group(self.grammar.array).setResultsName(
            'value').setParseAction(array_parse)
        self.grammar.ascii_line = self.grammar.ascii_kw + self.grammar.keyword_sep + self.grammar.text_parameter.setResultsName(
            'value').setParseAction(text_parse)
        self.grammar.line = (_pp.Dict(_pp.Group(self.grammar.ascii_line)) | _pp.Dict(_pp.Group(self.grammar.normal_line)) |  _pp.Dict(_pp.Group(self.grammar.date))) + EOL
        self.grammar.param_grammar = _pp.OneOrMore(self.grammar.line)

    def to_file(self, outfile):
        with open(outfile, 'w') as fout:
            for key, par in self.par_dict:
                par_string = "{key}: \t {value} \n".format(key=key, value=par)
                fout.write(par_string)


class SlcParams(ParameterFile):
    def __init(self, file):
        super(self.__class__, self).init()
        string_format = _pp.OneOrMore(_pp.Word(_pp.alphanums))
        date_format = _pp.Word(_pp.nums, exact=4) + _pp.Word(_pp.nums, exact=2) + _pp.Word(_pp.nums, exact=2)
        float_format = _pp.Word(_pp.nums)
        # with open(file, 'r') as fin:

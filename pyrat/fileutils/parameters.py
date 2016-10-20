import pyparsing as _pp


class ParameterFile:
    def __init__(self):
        EOL = _pp.LineEnd().suppress()
        _pp.ParserElement.setDefaultWhitespaceChars(' \t')
        self.grammar = type('grammar', (object,), {})()
        # separator of keyword
        self.grammar.keyword_sep = _pp.Literal(':').suppress()
        self.grammar.text_parameter = _pp.Group(_pp.Combine(
            _pp.restOfLine()))
        # Definition of numbers
        self.grammar.base_units = _pp.Literal('dB') | _pp.Literal('s') | _pp.Literal('m') | _pp.Literal(
            'Hz') | _pp.Literal('degrees') | _pp.Literal('1')
        self.grammar.repeated_unit = (self.grammar.base_units + _pp.Optional('^' + _pp.Word('-123')))
        self.grammar.unit = _pp.Group(
            self.grammar.repeated_unit + _pp.Optional(_pp.ZeroOrMore('/' + self.grammar.repeated_unit)))
        self.grammar.float_re = _pp.Regex('[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?')
        self.grammar.int = _pp.Word(_pp.nums)
        self.grammar.number = (self.grammar.float_re | self.grammar.int)
        # Definition of array:
        array = _pp.Forward()
        array << (
        (self.grammar.number + array + self.grammar.unit) | (self.grammar.number + _pp.Optional(self.grammar.unit)))
        self.grammar.array = array
        # Date
        self.grammar.date = _pp.Literal('date').setResultsName('keyword') + self.grammar.keyword_sep + _pp.Group(
            _pp.Word(_pp.nums + '.').setResultsName('year') + _pp.Word(_pp.nums + '.').setResultsName(
                'month') + _pp.Word(_pp.nums + '.').setResultsName('day')).setResultsName('date')
        # Line with general text
        self.grammar.ascii_kw = (
        _pp.Literal('title') | _pp.Literal('sensor') | _pp.Literal('image_format') | _pp.Literal(
            'image_geometry') | _pp.Literal('azimuth_deskew') | _pp.Literal('GPRI_TX_mode') |  _pp.Literal('GPRI_TX_antenna')).setResultsName('keyword')
        self.grammar.ascii_line = self.grammar.ascii_kw + self.grammar.keyword_sep + self.grammar.text_parameter
        # normal keyword
        self.grammar.normal_kw = ~_pp.Literal('date') + ~self.grammar.ascii_kw + _pp.Word(_pp.alphanums + '_')
        self.grammar.normal_line = self.grammar.normal_kw.setResultsName(
            'keyword') + self.grammar.keyword_sep + _pp.Group(self.grammar.array)
        self.grammar.line = (self.grammar.normal_line | self.grammar.ascii_line | self.grammar.date) + EOL
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

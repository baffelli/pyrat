import unittest
import datetime as dt
from pyrat.fileutils.parsers import  FasterParser, ParameterParser, FastestParser

class TestParser(unittest.TestCase):

    def setUp(self):
        self.n = 10
        self.array = range(1,self.n)
        self.units = ["m^{i}".format(i=i) for i in self.array]
        self.parser = FastestParser()

    def testIntParsing(self):
        hf = "array: {arr}".format(arr=' '.join(map(str, self.array)))
        parsed = self.parser.parse(hf)
        print(parsed.asDict())
        self.assertEqual(parsed.asDict()['array']['value'] , list(self.array) )

    def testDateParsing(self):
        date = dt.datetime(year=2016, month=6, day=14, hour=13, minute=12, second=23, microsecond=23)
        hf = """date: {dt}""".format(dt=date)
        parsed = self.parser.parse(hf)
        pd = parsed.asDict()
        print(parsed.pprint())
        self.assertEqual(pd['date']['value'], date)

    def testSimpleParsing(self):
        hf ='a: 45'
        parsed = self.parser.parse(hf)

    def testTitleParsing(self):
        hf="""scewmo cuil\nr: 4\na: 5\n"""
        parsed = self.parser.parse(hf)
        print(parsed.asDict())

    def testMultilineTitle(self):
        hf="""This is\na long\ntitle \na:4"""
        parsed = self.parser.parse(hf)
        print(parsed.asDict())

    def testShortDateParsing(self):
        dt_str = dt.datetime(2015,8,3).date()
        hf="""date: {dt}\n""".format(dt=dt_str)
        parsed = self.parser.parse(hf)
        self.assertEqual(parsed.asDict()['date']['value'], dt_str)

    def testUnitParsing(self):
        hf = """array: {arr} {units}
        """.format(arr=' '.join(map(str, self.array)), units=' '.join(map(str, self.units)))
        parsed = self.parser.parse(hf)
        self.assertEqual(parsed.asDict()['array'] , self.units )


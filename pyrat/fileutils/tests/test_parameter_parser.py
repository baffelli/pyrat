import unittest
import datetime as dt
from pyrat.fileutils.parameters import ParameterParser


class TestParser(unittest.TestCase):

    def setUp(self):
        self.n = 10
        self.array = range(1,self.n)
        self.units = ["m^{i}".format(i=i) for i in self.array]
        self.parser = ParameterParser()

    def testIntParsing(self):
        hf = """ffs st\narray: {arr}
        """.format(arr=' '.join(map(str, self.array)))
        parsed = self.parser.parse(hf)
        print(parsed)
        self.assertEqual(parsed.asDict()['array']['value'] , list(self.array) )

    def testDateParsing(self):
        date = dt.datetime(year=2016, month=6, day=14, hour=13, minute=12, second=23, microsecond=23)
        hf = """
        date: {dt}\n""".format(dt=date)
        parsed = self.parser.parse(hf)
        pd = parsed.asDict()
        print(pd)
        self.assertEqual(pd['date']['value'], date)

    def testUnitParsing(self):
        hf = """array: {arr} {units}
        """.format(arr=' '.join(map(str, self.array)), units=' '.join(map(str, self.units)))
        parsed = self.parser.parse(hf)
        print(hf)
        print(parsed)
        print(parsed['array'])
        self.assertEqual(parsed.asDict()['array']['unit'] , self.units )


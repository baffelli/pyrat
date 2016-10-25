import unittest
from pyrat.fileutils.parameters import ParameterParser


class TestParser(unittest.TestCase):

    def setUp(self):
        self.n = 10
        self.array = range(1,self.n)
        self.units = ["m^{i}".format(i=i) for i in self.array]
        self.parser = ParameterParser()

    def testIntParsing(self):
        hf = """
        array: {arr}
        """.format(arr=' '.join(map(str, self.array)))
        parsed = self.parser.parse(hf)
        self.assertEqual(parsed['array']['value'] , list(self.array) )

    def testUnitParsing(self):
        hf = """array: {arr} {units}
        """.format(arr=' '.join(map(str, self.array)), units=' '.join(map(str, self.units)))
        parsed = self.parser.parse(hf)
        print(hf)
        print(parsed)
        print(parsed['array'])
        self.assertEqual(parsed.asDict()['array']['unit'] , self.units )


import csv as csv
from unittest import TestCase
import glob as glob

from .. import utils

class TestListOfDates(TestCase):
    def setUp(self):
        with open('./list_of_slcs.csv') as infile:
            # reader = csv.reader(infile)
            dates = infile.readlines()
            dates = [date.strip() for date in dates]
        self.dates = dates
        self.list_of_dates = utils.ListOfDates(self.dates)

    def test_combinations(self):
        pass

    def testContains(self):
        a = '20150811_155521' in self.list_of_dates
        print(a)


    def test_select_manual_range(self):
        start = '20150803_000000'
        stop = '20150803_180000'
        dates = self.list_of_dates.select_date_range(start, stop)
        print(dates)


    def test_select_n_dates(self):
        res = self.list_of_dates.select_n_dates(self.dates[0], 1)
        print(res)

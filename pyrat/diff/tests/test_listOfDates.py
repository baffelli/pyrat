import csv as csv
from unittest import TestCase

from .. import utils


class TestListOfDates(TestCase):
    def setUp(self):
        with open('./data/list_of_slcs.csv') as infile:
            reader = csv.reader(infile, delimiter=',')
            dates = list(reader)[0]
        self.dates = dates
        self.list_of_dates = utils.ListOfDates(self.dates)

    def test_select_date_range(self):
        len = 4
        res = self.list_of_dates.select_date_range(self.dates[0], self.dates[len])
        print(res)
        self.assertEqual(res, self.dates[0:len + 1])

    def test_select_manual_range(self):
        start = '20150710_000000'
        stop = '20150710_180000'
        dates = self.list_of_dates.select_date_range(start, stop)
        print(dates)

    def test_select_n_dates(self):
        len = 4
        print(self.dates[0:5])

        res = self.list_of_dates.select_n_dates(self.dates[0], 1)
        print(res)

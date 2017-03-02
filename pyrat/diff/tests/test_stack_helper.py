import csv as csv
from unittest import TestCase
import glob as glob

from .. import utils


class TestStack(TestCase):

    def setUp(self):
        stack = utils.StackHelper.fromfile('./list_of_slcs.csv')
        self.stack = stack

    def test_combinations(self):
        start = '20150803_000000'
        stop = '20150803_180000'
        s = self.stack.all_combinations_with_separation(start, stop)

    def test_select_manual_range(self):
        start = '20150803_000000'
        stop = '20150803_180000'
        dates = self.list_of_dates.select_date_range(start, stop)
        print(dates)

    def test_select_n_dates(self):
        res = self.list_of_dates.select_n_dates(self.dates[0], 1)
        print(res)

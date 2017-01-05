from unittest import TestCase

import csv as csv
from .. import utils

class TestListOfDates(TestCase):

    def setUp(self):

        with open('./data/list_of_slcs.csv') as infile:
            reader = csv.reader(infile)
            dates = list(reader)
        self.dates = dates
        self.list_of_dates = utils.ListOfDates(self.dates)

    def test_select_date_range(self):
        self.fail()

    def test_select_n_dates(self):
        self.fail()

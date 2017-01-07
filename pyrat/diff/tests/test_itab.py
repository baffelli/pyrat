from unittest import TestCase
from .. import utils

class TestItab(TestCase):

    def setUp(self):
        self.nslc = 5
        self.file = './data/itab'
        self.itab = utils.Itab(self.nslc)


    def test_tofile(self):
        self.itab.tofile(self.file)

    def test_fromfile(self):
        itab = utils.Itab.fromfile(self.file)
        self.assertTrue(self.itab.n_slc == itab.n_slc)

    def test_to_incidence_matrix(self):
        A = self.itab.to_incidence_matrix()
        print(A)
        print(A.shape)

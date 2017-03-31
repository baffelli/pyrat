from unittest import TestCase
from .. import utils

class TestItab(TestCase):


    def testTimeSeries(self):
        tab = utils.Itab(8,step=1,stride=1)

    def testStride(self):
        tab = utils.Itab(12, step=2, stride=2)
        print(tab)

    def test_tofile(self):
        self.itab.tofile(self.file)

    def test_fromfile(self):
        itab = utils.Itab.fromfile(self.file)
        self.assertTrue(self.itab.n_slc == itab.n_slc)

    def test_to_incidence_matrix(self):
        A = self.itab.to_incidence_matrix()
        print(A)
        print(A.shape)

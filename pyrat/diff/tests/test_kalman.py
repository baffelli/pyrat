import unittest
from ..kalman import KalmanFilter
from .. import kalman as ka
import numpy as np


class TestKalman(unittest.TestCase):

    def testMatrixVectorProduct(self):
        npixels = 150
        nstates = 2
        F = np.tile(np.eye(nstates), (npixels, 1, 1))
        x0 = np.tile(np.zeros(nstates), (npixels, 1))
        x_t = ka.matrix_vector_product(F, x0)
        self.assertEqual(x0, x_t)

    def testVectorFilter(self):
        npixels = 1500
        nsteps = 10
        nstates = 2
        noutputs = 2
        F = np.tile(np.eye(nstates), (npixels, nsteps,1,1))
        H = np.tile(np.eye(noutputs),(npixels, nsteps,1,1))
        R = np.tile(np.eye(noutputs),(npixels,1,1))
        z = np.tile(np.zeros(noutputs),(npixels, nsteps,1,))
        x0 = np.tile(np.zeros(nstates),(npixels,1))
        Q = np.tile(np.eye(nstates),(npixels,1,1))
        ka = KalmanFilter(F=F, H=np.eye(2), R=R, x_0=x0, Q=Q)
        ka.predict()
        ka.update(z)
        self.assertEqual(Q.shape, ka.P.shape)


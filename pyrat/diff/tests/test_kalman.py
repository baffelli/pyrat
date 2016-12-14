import unittest
from ..kalman import KalmanFilter
import numpy as np


class TestKalman(unittest.TestCase):

    def testVectorFilter(self):
        npixels = 15
        nstates = 2
        noutputs = 2
        F = np.tile(np.eye(nstates), (npixels,1,1))
        H = np.tile(np.eye(noutputs),(npixels,1,1))
        R = np.tile(np.eye(noutputs),(npixels,1,1))
        z = np.tile(np.zeros(noutputs),(npixels,1,))
        x0 = np.tile(np.zeros(nstates),(npixels,1))
        Q = np.tile(np.eye(nstates),(npixels,1,1))
        ka = KalmanFilter(nstates=nstates, noutpus=noutputs, F=F, H=np.eye(2), R=R, x0=x0, Q=Q)
        ka.predict()
        self.assertEqual(Q.shape, ka.P.shape)
        ka.update(z)


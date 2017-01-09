import unittest

import matplotlib.pyplot as plt
import numpy as np

from . import test_data
from .. import kalman as ka
from ..kalman import KalmanFilter


class TestKalman(unittest.TestCase):

    def setUp(self):
        #Set seed for PRNG
        self.seed = 1234567890
        #Load data and generate reference filter and samples
        self.uniform_data = test_data.UniformMotion()
        self.x_0 = [self.uniform_data.initial_position, self.uniform_data.initial_velocity]
        self.reference_filter = ka.KalmanFilter(F=self.uniform_data.F, H=self.uniform_data.H, R=self.uniform_data.R,
                                         Q=self.uniform_data.Q, x_0=self.x_0, P_0=self.uniform_data.P_0)
        #generate samples using seed
        nsamples = 1000
        self.x_sampled, self.z = self.reference_filter.sample(nsamples,self.x_0, seed=self.seed)



    def testMatrixVectorProduct(self):
        """
        Test if the tensor matrix-vector multiplication
        works as expected
        Returns
        -------

        """
        npixels = 150
        nstates = 2
        F = np.tile(np.eye(nstates), (npixels, 1, 1))
        x0 = np.tile(np.ones(nstates), (npixels, 1))
        x_t = ka.matrix_vector_product(F, x0)
        self.assertTrue(np.allclose(x0, x_t))

    def testSampling(self):
        #Geenrate samples with another seed
        states, observations = self.reference_filter.sample(1000, x_0=self.x_0, seed=24)
        t = np.arange(observations.shape[0])
        #Plot them to check
        f, (x_ax, v_ax) = plt.subplots(2, 1, sharex=True)
        x_ax.plot(states[:, 0, 0])
        x_ax.plot(self.x_sampled[:, 0, 0])
        v_ax.plot(states[:, 0, 1])
        v_ax.plot(self.x_sampled[:, 0, 1])
        v_ax.xaxis.set_label_text('sample index')
        plt.show()

    def testFilter(self):
        x_sm, P_sm = self.reference_filter.filter(self.z)
        f, (x_ax, v_ax) = plt.subplots(2, 1, sharex=True)
        x_ax.plot(x_sm[:, 0, 0])
        x_ax.plot(self.x_sampled[:, 0, 0])
        v_ax.plot(x_sm[:, 0, 1])
        v_ax.plot(self.x_sampled[:, 0, 1])
        v_ax.xaxis.set_label_text('sample index')
        plt.show()

    def testSmooth(self):
        x_sm, P_sm, L = self.reference_filter.smooth(self.z)
        print(x_sm.shape)
        f, (x_ax, v_ax) = plt.subplots(2, 1, sharex=True)
        x_ax.plot(x_sm[:, 0, 0])
        x_ax.plot(self.x_sampled[:, 0, 0])
        v_ax.plot(x_sm[:, 0, 1])
        v_ax.plot(self.x_sampled[:, 0, 1])
        v_ax.xaxis.set_label_text('sample index')
        plt.show()


    def testRealData(self):
        #load inputs
        z = np.load('./data/Z.npy')
        # load  transition matrices
        F = np.array(np.load('./data/F.npy'), ndmin=4).swapaxes(0,1)
        # load output matrices
        H = np.array(np.load('./data/H.npy'),ndmin=4).swapaxes(0,1)
        x_0=np.zeros((F.shape[1], F.shape[2]))
        P_0=np.tile(np.eye(F.shape[2]),(F.shape[1],1,1))
        filter = ka.KalmanFilter(H=H, F=F, x_0=x_0, P_0=P_0)
        x_filt, P_filt, L  = filter.smooth(z)
        plt.plot(x_filt[:,220,1])
        plt.show()

    def testEMR(self):


        #Smooth
        x_s, P_s, L = self.reference_filter.smooth(self.z)
        t = np.arange(x_s.shape[0])

        #Plot observations
        # f, (x_ax, v_ax) = plt.subplots(2, 1, sharex=True)
        # x_ax.plot(self.z[:, 0, 0])
        # ka.plot_state_and_variance(x_ax, t, x_s[:, 0, 0], P_s[:, 0, 0, 0])
        # x_ax.plot(x_s[:, 0, 0])
        # v_ax.plot(self.z[:, 0, 1])
        # ka.plot_state_and_variance(v_ax, t, x_s[:, 0, 1], P_s[:, 0, 1, 1])
        # plt.show()
        # Setup second filter with unknown matrices R and Q
        Q = np.eye(2) * 1e-4
        em_filt = ka.KalmanFilter(F=self.uniform_data.F, H=self.uniform_data.H, Q=Q)
        # Run EM
        em_filt.EM(self.z[0:50], ['R'], niter=20)
        print(em_filt.R)


    def testEMF(self):


        #Smooth
        x_s, P_s, L = self.reference_filter.smooth(self.z)
        t = np.arange(x_s.shape[0])

        #Plot observations
        # f, (x_ax, v_ax) = plt.subplots(2, 1, sharex=True)
        # x_ax.plot(self.z[:, 0, 0])
        # ka.plot_state_and_variance(x_ax, t, x_s[:, 0, 0], P_s[:, 0, 0, 0])
        # x_ax.plot(x_s[:, 0, 0])
        # v_ax.plot(self.z[:, 0, 1])
        # ka.plot_state_and_variance(v_ax, t, x_s[:, 0, 1], P_s[:, 0, 1, 1])
        # plt.show()
        # Setup second filter with unknown matrices R and Q
        Q = np.eye(2) * 1e-4
        em_filt = ka.KalmanFilter(F=None, H=self.uniform_data.H, Q=self.uniform_data.Q, R=self.uniform_data.R)
        # Run EM
        em_filt.EM(self.z[0:50], ['F'], niter=20)
        print(em_filt.F)



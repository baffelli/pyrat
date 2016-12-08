import numpy as np
import gdal as _gd
import queue

import pyrat.diff.intfun

from .. ipt import core as ipt

def special_inv(M):
    if np.isscalar(M):
        return 1 / M
    else:
        return np.linalg.pinv(M)


def noise_fun(m, sigma):
    if np.isscalar(sigma):
       return m + np.random.randn(*m.shape) * np.sqrt(sigma)
    else:
        return np.random.multivariate_normal(m, sigma)

class LinearSystem:
    """
    Simple class to implement a linear system
    """
    def __init__(self, F=None, H=None, x0=None, B=0, Q=None, R=None):
        self.x = x0
        self.x_noisy = noise_fun(self.x, Q)
        self.z = noise_fun(np.dot(H, self.x), R)
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        if B:
            self.B = B

    def state_transition(self):
        """
        Computes the next state given the current input
        Parameters
        ----------
        x

        Returns
        -------

        """
        self.x = np.dot(self.F, self.x)
        self.x_noisy = noise_fun(np.dot(self.F, self.x_noisy), self.Q)
        return self.x

    def output(self):
        # self.z = np.dot(self.H, self.x)
        self.z = noise_fun(np.dot(self.H, self.x_noisy), self.R)
        return self.z




class InterferogramStackSimulator:
    def __init__(self, model ,stride=1, step=1, window=5, lam=1.7e-2):
        self.window = window#number of interferograms in stack
        self.stride = stride#increment of master
        self.step = step#increment of slave
        self.itab = pyrat.diff.intfun.itab()
        self.lam = lam #wavelength
        self.x = []
        self.model = model
        self.x_memory = queue.Queue(maxsize=self.window)
        self.z = np.zeros(self.nstack * self.model.x.shape[0])

    def predict(self):
        #remove oldest
        next_state = self.model.state_transition()
        self.x.append(next_state)
        try:
            self.x_memory.put_nowait(next_state)
        except:
            pass


    def output(self):
        reference = self.x_memory.get()
        memory = list(self.x_memory.queue)
        dx = [reference - x for x in memory ]
        return dx


class KalmanFilter:
    def __init__(self, nstates, noutpus, ninputs=0, F=None, B=None, H=None, R=None, Q=None, x0=None):



        self.nstates = nstates
        self.ninputs = ninputs
        self.noutputs = noutpus

        #Initial state of filter
        if x0 is not None:
            self._x = x0
        else:
            self._x = np.zeros(nstates)

        # State transition
        if F is not None:
            self.F = F
        else:
            self.F = np.eye(self.nstates)
        # Control matrix
        if B is not None:
            self.B = B
        elif B is None:
            self.B = np.zeros((self.nstates,)) if self.ninputs == 0 or self.ninputs == 1 else np.zeros(
                (self.nstates, self.ninputs))
        # Output
        if H is not None:
            self.H = H
        else:
            self.H = np.eye((self.noutputs, self.nstates))
        # System uncertainity
        if Q is not None:
            self.Q = Q
        else:
            self.Q = np.eye(self.nstates)
        # Measureement covariance
        if R is not None:
            self.R = R
        else:
            self.R = np.eye(self.noutputs)

        # State estimate covariance
        self._P = np.eye(nstates)

    def innovation(self, z):
        return z - np.dot(self.H, self._x)

    def predict(self, u=0, F=None, B=None, Q=None):
        """
        Predict next filter state and its covariance
        Parameters
        ----------
        u
        F
        B
        Q

        Returns
        -------

        """
        B = B or self._B
        F = F or self._F
        Q = Q or self._Q
        # Compute next state
        self._x = np.dot(F, self.x) + np.dot(B, u)
        # Compute state covariance
        self._P = F.dot(self._P).dot(F.T.conj()) + Q

    def update(self, z, R=None, H=None):

        R = R or self.R
        H = H or self.H

        # Innovation
        self._y = z - np.dot(H, self.x)
        # Residual covariance
        S = H.dot(self._P).dot(H.T.conj()) + R
        # Kalman gain
        K = self._P.dot(H.T.conj()).dot(special_inv(S))
        self._x = self.x + np.dot(K, self._y)
        self._P = (np.eye(self.nstates) - np.dot(K, H)).dot(self._P)

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, value):
        if np.isscalar(value) and self.nstates == 1:
            self._F = value
        elif value.shape[0] == value.shape[1] == self.nstates:
            self._F = value
        else:
            raise Exception("F is not of shape {}X{} or scalar".format(self.nstates, self.nstates))

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, value):
        if np.isscalar(value) and self.nstates == 1:
            self._B = value
        elif value.shape[0] == self.nstates and self.ninputs < 2:
            self._B = value
        elif value.shape[0] == self.nstates and value.ndim > 1 and value.shape[1] == self.ninputs:
            self._B = value
        else:
            raise Exception("B is not of shape {}X{} or scalar".format(self.nstates, self.ninputs))

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, value):
        if np.isscalar(value) and self.nstates == 1 and self.noutputs == 1:
            self._H = value
        elif value.shape[0] == self.nstates and self.noutputs == 1:
            self._H = value
        elif value.shape[0] == self.noutputs and value.shape[1] == self.nstates:
            self._H = value
        else:
            raise Exception("H is not of shape {}X{} or scalar".format(self.noutputs, self.nstates))

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, value):
        if np.isscalar(value) and self.nstates == 1:
            self._Q = value
        elif value.shape[0] == value.shape[1] == self.nstates:
            try:  # only accept positive definite matrices
                np.linalg.cholesky(value)
            except np.LinAlgError:
                raise np.LinAlgError("Q is not positive definite, cannot be used as a state covariance matrix")
            self._Q = value
        else:
            raise np.LinAlgError("Q is not positive definite, cannot be used as a state covariance matrix")


    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        if np.isscalar(value) and self.nstates == 1 and self.noutputs == 1:
            self._Q = value
        elif value.shape[0] == self.noutputs and self.noutputs == 1:
            self._Q = value
        elif value.shape[0] == self.noutputs and value.shape[1] == self.noutputs and self.noutputs > 1:
            try:  # only accept positive definite matrices
                np.linalg.cholesky(value)
            except np.LinAlgError:
                raise np.LinAlgError("R is not positive definite, cannot be used as a state covariance matrix")
            self._R = value
        else:
            raise TypeError("R is not of shape {}X{} or scalar".format(self.noutputs, self.noutputs))


    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        if np.isscalar(value) and self.nstates == 1 or value.shape[0] == self.nstates:
            self._x = value

    @property
    def y(self):
        return np.dot(self.H, self.x)



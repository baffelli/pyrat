# Module for kalman filtering in python. Freely based on pykalman

import pickle

import numpy as np


def special_inv(M):
    if np.isscalar(M):
        return 1 / M
    else:
        return np.linalg.inv(M)


def noise_fun(m, sigma):
    if np.isscalar(sigma):
        return m + np.random.randn(*m.shape) * np.sqrt(sigma)
    else:
        return np.random.multivariate_normal(m, sigma)


def avoid_none(var, alternative_var):
    if var is None:
        return alternative_var
    else:
        return var


def isPSD(A, tol=1e-6):
    """
    Check if the matrix is positive semidefinite
    to a certain tolerance
    Parameters
    ----------
    A
    tol

    Returns
    -------
    """
    E = np.linalg.eigvalsh(A)
    if np.all(E > -tol):
        pass
    else:
        raise np.linalg.LinAlgError('A is not positve semidefinite')


def matrix_vector_product(A, x):
    prod = np.einsum('...ij,...j->...i', A, x)
    return prod


def matrix_matrix_product(A, B):
    return np.einsum('...ij,...jk->...ik', A, B)


def transpose_tensor(A):
    if A.ndim <= 2:
        return A.T
    else:
        return np.einsum('...ij->...ji', A)


def special_eye(A):
    if A.ndim == 2:
        return np.eye(A.shape[0])
    else:
        return np.tile(np.eye(A.shape[-1]), (A.shape[0], 1, 1))


def shape_or_default(arr, index):
    """
    Get shape of array or 1 if
    the object does not support shape
    Parameters
    ----------
    arr
    index

    Returns
    -------
        `a.shape[index]` or 1
    """
    try:
        return arr.shape[index]
    except AttributeError:
        return 1


def get_observations_shape(observations):
    if observations.ndim == 2:
        return 1, observations.shape[0]
    else:
        return observations.shape[0], observations.shape[1]


def pick_nth_step(matrix, index, ndims=2):
    """
    Picks the matrix/vector corresponding to the
    n-th filter step
    Parameters
    ----------
    matrix
    index
    ndims

    Returns
    -------

    """
    if matrix.ndim >= ndims + 1:
        return matrix[index]
    elif matrix.ndim == ndims:
        return matrix
    else:
        raise IndexError


def kalman_prediction_step(x, P, F, Q):
    # control_input = matrix_vector_product(B, u)
    x_predicted = matrix_vector_product(F, x)
    P_predicted = matrix_matrix_product(matrix_matrix_product(F, P), transpose_tensor(F).conj()) + Q
    return x_predicted, P_predicted


def kalman_update_step(x, P, F, z, H, R):
    # Innovation
    y = z - matrix_vector_product(H, x)
    # Residual covariance
    S = matrix_matrix_product(matrix_matrix_product(H, P), transpose_tensor(H).conj()) + R
    # Kalman gain
    K = matrix_matrix_product(matrix_matrix_product(P, transpose_tensor(H).conj()), special_inv(S))
    x_update = x + matrix_vector_product(K, y)
    P_update = matrix_matrix_product((special_eye(P) - matrix_matrix_product(K, H)), P)
    return x_update, P_update, K


def kalman_smoothing_step(F, x_filtered, P_filtered, x_predicted, P_predicted, x_smooth, P_smooth)
    """
    Run kalman smoothing step to
    obtain posterior state distribution given all observations.

    Parameters
    ----------
    x_filtered : array-like
        Posterior state mean at time t given observations from times 0 to t. Must have shape `(nmatrices,nstates)`.
    P_filtered :  array-like
        Posterior state covariance at time t given observation from times 0 to t. Has shape `(nmatrices,nstates,nstates)`
    x_predicted : array-like
        Prior state mean at time t+1 given observation from times 0 to t, with shape `(nmatrices,nstates,nstates)`
    P_predicted : (nmatrices, nstates, nstates) array-like
        Prior state covariance at time t+1 given observation from times 0 to t
    F : array-like
        State stransition matrix from t to t+1, with shape  `(nmatrices, nstates, nstates)`
    x_smooth : array-like
        Posterior state mean at time t+1 given all observations from  times 0 to ntimesteps, shape `(nmatrixes, nstates)`
    P_smooth : array-like
        Posterior state covariance at time t+1 given all observations from  times 0 to ntimesteps, array with shape `(nmatrices, nstates, nstates)`

    Returns
    -------
    x_smooth :  array-like
        Posterior state mean at time t given all observations from times 0 to ntimesteps-1, stored in an `numpy.ndarray` of shape `(nmatrices,nstates)`
    P_smooth :  array-like
        Posterior state covariance at time t given all observations from times 0 to ntimesteps-1, stored as `(nmatrices,nstates,nstates)`
    L :  array-like
        Kalman smoothing gain at time t, ordered as `(nmatrices,nstates,nstates)`
    """
    # "Kalman-like matrix to include predicted"
    L = matrix_matrix_product(P_filtered, transpose_tensor(F), special_inv(P_predicted))
    x_smooth = x_filtered + matrix_matrix_product(L, (x_smooth - x_predicted))
    P_smooth = P_filtered + matrix_matrix_product(matrix_matrix_product(L, (P_smooth - P_predicted)),
                                                  transpose_tensor(L))
    return x_smooth, P_smooth, L


def filter(F, Q, H, R, x_0, P_0, z):
    """
    Run Kalman Filter to estimate posterior distribution given all observation up to the current timestep

    Parameters
    ----------
    F : array-like
        State transition matrix wit shapes  `(ntimesteps, nmatrices, nstates, nstates)` or `(nmatrices, nstates, nstates)`
    Q :  array-like
        State transition covariance ("model uncertainity")
    H : (ntimesteps, nmatrices, noutputs, nstates) or (nmatrices, noutputs, nstates) array-like
        Observation matrix
    R : (ntimesteps, nmatrices, noutputs, noutputs) or (nmatrices, noutputs, nstates) array-like
        Measurement covariance matrix
    x_0 : (nmatrices, nstates) array-like
        Initial state mean
    P_0 : (nmatrices, nstates, nstates) array-like
        Initial state covariance matrix
    z   :  (ntimesteps, nmatrices , noutputs) array-like
        Observation from 0 to ntimesteps - 1

    Returns
    -------
    x_predicted
    P_predicted
    K


    """
    ntimesteps, nmatrices = get_observations_shape(observations)
    x_predicted = np.zeros((ntimesteps, nmatrices, self.nstates))
    P_predicted = np.zeros((ntimesteps, nmatrices, self.nstates, self.nstates))
    K = np.zeros((ntimesteps, nmatrices, self.nstates, self.noutputs))
    x_filtered = x_predicted * 0
    P_filtered = P_predicted * 0
    for t in range(ntimesteps):
        if t == 0:
            x_predicted[0, :, :] = self.x0
            P_predicted[0, :, :] = self.P
        else:
            F = pick_nth_step(self.F, t)
            H = pick_nth_step(self.H, t)
            z_cur = pick_nth_step(observations, t, ndims=1)
            x_predicted[t], P_predicted[t] = kalman_prediction_step(x_filtered[t - 1], P_filtered[t - 1], F,
                                                                    self.Q)  # predict
            x_filtered[t], P_filtered[t], K[t] = kalman_update_step(x_predicted[t], P_predicted[t], F, z_cur, H,
                                                                    self.R)  # update
    return x_predicted, P_predicted, K, x_filtered, P_filtered

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


class KalmanFilter:
    """
    This class implements the Kalman Filter and Smoother, supports the computation
    of the filter simultaneously on a group of matrices, eg when using it for computer vision
    applications, where the filter is simultaneously run on each pixel, possibly with different matrices. In the following, the matrices are supposed to be "stacked"
    along  the first dimension.

    Parameters
    ----------
    F : (ntimesteps, nmatrices, nstates, nstates) or (nstates, nstates) array-like
        State transition matrix from :math:`t` to :math:`t+1`. Can be a sequence of matrices of length `ntimesteps` if the state
        transition matrix varies over time.
    Q : (nmatrices, nstates, nstates) array-like
        Transition covariance (model uncertainity) matrix for the system.
    H : (ntimesteps, nmatrices, noutputs, nstates) or (nmatrices, noutputs, nstates) array-like
        Observation matrix to compute observation from state.
    R : (nmatrices, noutputs, noutputs) array-like
        Observation covariance matrix
    x0 : (nmatrices, nstates) array-like optional
        Initial state mean
    P : (nmatrices, nstates, nstates)
        Initial state covariance
    """

    def __init__(self, ninputs=0, F=None, B=None, H=None, R=None, Q=None, x0=None,
                 P=None):

        # Determine state space size from last dimension of F
        self.nstates = shape_or_default(F, -1)
        self.ninputs = ninputs
        # Determine the number of outpit from second to last dimension of H
        self.noutputs = shape_or_default(H, -2)

        # Initial state of filter
        if x0 is not None:
            self.x0 = np.array(x0)
        else:
            self.x0 = np.zeros(self.nstates)

        # State transition
        if F is not None:
            self.F = np.array(F)
        else:
            self.F = np.eye(self.nstates)
        # Control matrix
        if B is not None:
            self.B = np.array(B)
        elif B is None:
            self.B = np.zeros((self.nstates,)) if self.ninputs == 0 or self.ninputs == 1 else np.zeros(
                (self.nstates, self.ninputs))
        # Output
        if H is not None:
            self.H = np.array(H)
        else:
            self.H = np.eye((self.noutputs, self.nstates))
        # System uncertainity
        if Q is not None:
            self.Q = np.array(Q)
        else:
            self.Q = np.eye(self.nstates)
        # Measureement covariance
        if R is not None:
            self.R = np.array(R)
        else:
            self.R = np.eye(self.noutputs)

        # State estimate covariance
        if P is not None:
            self.P = np.array(P)
        else:
            self.P = np.eye(self.nstates)



    def smooth(self, z):
        """
        Runs the Kalman smoother, estimates the state for each time given all observations
        Parameters
        ----------
        z : (ntimesteps, nmatrices, ninputs) array-like
            All observations from 0 to ntimesteps

        Returns
        -------
        x_smooth : (ntimesteps, nmatrices, nstates) array_like
            The posterior smoothed state for each time from 0 to ntimesteps, computed using all observations
        x_smooth: (ntimesteps, nmatrices, nstates, nstates)
            The posterior smoothed state covariance for each time from 0 to ntimesteps, computed using all observations
        L : (ntimesteps, nmatrices, nstates, nstates) array-like
            The Kalman correction matrix for all steps
        """
        ntimesteps, nmatrices = get_observations_shape(z)
        x_smooth = np.zeros((ntimesteps, nmatrices, self.nstates))
        P_smooth = np.zeros((ntimesteps, nmatrices, self.nstates, self.nstates))
        L = np.zeros((ntimesteps - 1, nmatrices, self.nstates, self.nstates))
        # First run kalman filter
        x_predicted, P_predicted, K, x_filtered, P_filtered = self.filter(z)
        #set mean and covariance at the end to the  forward filtered data to start the smoother
        x_smooth[-1] = x_filtered[-1]
        P_smooth[-1] = P_filtered[-1]
        #Run the smoother backwards
        for t in reversed(range(ntimesteps - 1)):
            F = pick_nth_step(self.F, t)
            x_smooth[t], P_smooth[t], L[t] = kalman_smoothing_step(F, x_filtered[t], P_filtered[t], x_predicted[t + 1],
                                                                   P_predicted[t+1], x_smooth[t+1], P_smooth[t+1])
        return x_smooth, P_smooth, L


    def em_observation_covariance(self, observations, transition_matrices):
        ntimesteps, nmatrices = get_observations_shape(observations)
        for t in range(ntimesteps):
            F = pick_nth_step(transition_matrices, t)



    def output(self, H=None):
        # System output
        H = avoid_none(H, self.H)
        return matrix_vector_product(H, self.x)

    def innovation(self, z, H=None):
        # Residual: measurement - output
        H = avoid_none(H, self.H)
        return z - self.output(H=H)

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
        # B = avoid_none(B, self.B)
        F = avoid_none(F, self.F)
        Q = avoid_none(Q, self.Q)
        # Compute next state
        if B is not None:
            control_input = matrix_vector_product(B, u)
        else:
            control_input = np.zeros(self.x.shape)
        self.x = matrix_vector_product(F, self.x) + control_input
        # Compute state covariance
        self.P = matrix_matrix_product(matrix_matrix_product(self.F, self.P), transpose_tensor(F).conj()) + Q
        # self.P = F.tensordot(self.P).tensordot(F.T.conj()) + Q

    def update(self, z, R=None, H=None):

        R = avoid_none(R, self.R)
        H = avoid_none(H, self.H)

        # Innovation
        y = self.innovation(z, H=H)
        # Residual covariance
        S = matrix_matrix_product(matrix_matrix_product(H, self.P), transpose_tensor(H).conj()) + R
        # Kalman gain
        K = matrix_matrix_product(matrix_matrix_product(self.P, transpose_tensor(H).conj()), special_inv(S))
        # K = self.P.dot(H.T.conj()).dot(special_inv(S))
        self.x = self.x + matrix_vector_product(K, y)
        P_new = matrix_matrix_product((special_eye(self.P) - matrix_matrix_product(K, H)), self.P)
        self.P = P_new

    def tofile(self, file):
        """
        Saves current kalman filter state to file
        Parameters
        ----------
        file

        Returns
        -------

        """
        with open(file, 'wb') as fp:
            pickle.dump(self, fp)

    def fromfile(cls, file):
        """
        Saves current kalman filter state to file
        Parameters
        ----------
        file

        Returns
        -------

        """
        with open(file, 'rb') as fp:
            return pickle.load(fp)

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, value):
        if np.isscalar(value) and self.nstates == 1:
            self._F = value
        elif value.shape[-2] == value.shape[-1] == self.nstates:
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
        elif value.shape[-1] == self.nstates and self.ninputs < 2:
            self._B = value
        elif value.shape[-1] == self.nstates and value.ndim > 1 and value.shape[-1] == self.ninputs:
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
        elif value.shape[-1] == self.nstates and self.noutputs == 1:
            self._H = value
        elif value.shape[-2] == self.noutputs and value.shape[-1] == self.nstates:
            self._H = value
        else:
            raise Exception("H is not of shape {}X{} or scalar".format(self.noutputs, self.nstates))

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, value):
        if np.isscalar(value) and self.nstates == 1:
            self._P = value
        elif value.shape[-2] == value.shape[-1] == self.nstates:
            try:  # only accept positive definite matrices
                isPSD(value)
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError("P is not positive definite, cannot be used as a prior covariance matrix")
            self._P = value
        else:
            raise np.linalg.LinAlgError("P is not positive definite, cannot be used as a prior covariance matrix")

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, value):
        if np.isscalar(value) and self.nstates == 1:
            self._Q = value
        elif value.shape[-2] == value.shape[-1] == self.nstates:
            try:  # only accept positive definite matrices
                isPSD(value)
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError("Q is not positive definite, cannot be used as a state covariance matrix")
            self._Q = value
        else:
            raise np.linalg.LinAlgError("Q is not positive definite, cannot be used as a state covariance matrix")

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, value):
        if np.isscalar(value) and self.nstates == 1 and self.noutputs == 1:
            self._Q = value
        elif value.shape[-2] == self.noutputs and self.noutputs == 1:
            self._Q = value
        elif value.shape[-2] == self.noutputs and value.shape[-1] == self.noutputs and self.noutputs > 1:
            try:  # only accept positive definite matrices
                isPSD(value)
            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError("R is not positive definite, cannot be used as a state covariance matrix")
            self._R = value
        else:
            raise TypeError("R is not of shape {}X{} or scalar".format(self.noutputs, self.noutputs))

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, value):
        if np.isscalar(value) and self.nstates == 1 or \
                        value.shape[-1] == self.nstates:
            self._x0 = value

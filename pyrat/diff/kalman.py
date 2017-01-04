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


def tensor_outer(A):
    return np.einsum('...i,...j->...ij', A, A)


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


def get_state_sequence_length(states):
    if states.ndim <= 2:
        return 1
    elif states.ndim == 3:
        return states.shape[0]


def get_state_size(F):
    if F.ndim > 1:
        return F.shape[-1]
    else:
        return 1


def get_output_size(H):
    if H.ndim > 1:
        return H.shape[-2]
    else:
        return 1


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


def kalman_prediction_step(F, x, P, Q):
    """
    Run kalman prediction step to estimate the prior state mean and covariance at time :math:`t+1` given the posterior mean and covariance at time :math:`t`.

    Parameters
    ----------
    x : array-like
        Posterior state mean at time :math:`t` given observations between 0 and :math:`t`
    P : array-like
        Posterior state covariance at time :math:`t` given observations between 0 and :math:`t`
    F : array-like
        State transition matrix between :math:`t` and :math:`t+1`
    Q : array-like
        State transition covariance between :math:`t` and :math:`t+1`

    Returns
    -------
    x_predicted : array-like
        Prior state mean at time :math:`t+1` given observations from 0 to :math:`t`
    P_predicted : array-like
        Prior state covariance at time :math:`t+1` given observations from 0 to :math:`t`

    """
    # control_input = matrix_vector_product(B, u)
    x_predicted = matrix_vector_product(F, x)
    P_predicted = matrix_matrix_product(matrix_matrix_product(F, P), transpose_tensor(F).conj()) + Q
    return x_predicted, P_predicted


def kalman_update_step(x_predicted, P_predicted, z, H, R):
    """
    Run Kalman update step to improve the predicted state using the current observation.

    Parameters
    ----------
    x_predicted : array-like
        Prior state mean at time :math:`t` given observations from times 0 to :math:`t-1`. Must have shape `(nmatrices,nstates)`
    P_predicted : array-like
         Prior state covariance matrix at time :math:`t` given observations from times 0 to :math:`t-1`. Must have shape `(nmatrices,nstates, nstates)`
    z : array-like
        Observation at time :math:`t`, with shape `(nmatrices,noutputs)`
    H : array-like
        Observation matrix at time :math:`t`,  has shape `(nmatrices,noutputs, nstates)`
    R : array-like
        Observation covariance matrix at time :math:`t`


    Returns
    -------
    x_updated : array-like
        Posterior state mean at time :math:`t` given observations from times 0 to :math:`t`.
    P_updated: array-like
        Posterior state covariance at time :math:`t` given observations from times 0 to :math:`t`.
    K : array-like
        Kalman gain matrix at time :math:`t`
    """
    # Innovation
    y = z - matrix_vector_product(H, x_predicted)
    # Residual covariance
    S = matrix_matrix_product(matrix_matrix_product(H, P_predicted), transpose_tensor(H).conj()) + R
    # Kalman gain
    K = matrix_matrix_product(matrix_matrix_product(P_predicted, transpose_tensor(H).conj()), special_inv(S))
    x_updated = x_predicted + matrix_vector_product(K, y)
    P_updated = matrix_matrix_product((special_eye(P_predicted) - matrix_matrix_product(K, H)), P_predicted)
    return x_updated, P_updated, K


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
        State transition matrix with shapes  `(ntimesteps, nmatrices, nstates, nstates)` or `(nmatrices, nstates, nstates)`, transition from :math:`t` to :math:`t+1` or from :math:`t_i` to :math:`t_{i + 1}` for time varying system.
    Q :  array-like
        State transition covariance ("model uncertainity") of shape `(ntimesteps-1, nmatrices, nstates, nstates)` or `(nmatrices, nstates, nstates)`
    H : array-like
        Observation matrix, shapes `(ntimesteps, nmatrices, noutputs, nstates)` or `(nmatrices, noutputs, nstates)`
    R :  array-like
        Measurement covariance matrix, `(ntimesteps, nmatrices, noutputs, noutputs)` or `(nmatrices, noutputs, nstates)`
    x_0 : array-like
        Initial state mean, shape `(nmatrices, nstates)`
    P_0 : (nmatrices, nstates, nstates) array-like
        Initial state covariance matrix, shape  `(nmatrices, nstates)`
    z   :  array-like
        Observation from 0 to ntimesteps - 1, in an array of shape `(ntimesteps, nmatrices , noutputs)`

    Returns
    -------
    x_predicted : array-like
        Prior state mean for all time steps from 0 to `ntimesteps`
    P_predicted : array-like
         Prior state covariance for all time steps from 0 to `ntimesteps`
    K : array-like
        Kalman weighting matrix for each timestep
    x_filtered : array-like
        Posterior state mean at each time :math:`t` given observations from 0 to to :math:`t`
    P_filtered : array-like
        Posterior state covariance at each time :math:`t` given observations from 0 to to :math:`t`


    """
    ntimesteps, nmatrices = get_observations_shape(z)
    nstates = get_state_size(F)
    noutputs = get_state_size(H)
    x_predicted = np.zeros((ntimesteps, nmatrices, nstates))
    P_predicted = np.zeros((ntimesteps, nmatrices, nstates, nstates))
    K = np.zeros((ntimesteps, nmatrices, nstates, noutputs))
    x_filtered = x_predicted * 0
    P_filtered = P_predicted * 0
    for t in range(ntimesteps):
        if t == 0:
            x_predicted[0, :, :] = x_0
            P_predicted[0, :, :] = P_0
        else:
            x_predicted[t], P_predicted[t] = kalman_prediction_step(F, x_filtered[t - 1], P_filtered[t - 1],
                                                                    Q)  # predict

        F = pick_nth_step(F, t)
        H = pick_nth_step(H, t)
        z_cur = pick_nth_step(z, t, ndims=1)
        x_filtered[t], P_filtered[t], K[t] = kalman_update_step(x_predicted[t], P_predicted[t], F, z_cur, H)  # update
    return x_predicted, P_predicted, K, x_filtered, P_filtered


def smooth(F, x_predicted, P_predicted, x_filtered, P_filtered, z):
    """
    Runs the Kalman smoother, estimates the state for each time given all observations

    Parameters
    ----------
    F : array-like
        State transition matrix with shapes  `(ntimesteps, nmatrices, nstates, nstates)` or `(nmatrices, nstates, nstates)`, transition from :math:`t` to :math:`t+1` or from :math:`t_i` to :math:`t_{i + 1}` for time varying system
    x_predicted : array-like
        Prior state means for all :math:`t` 0 to `ntimesteps`, given observations from 0 to the current time :math:`t`
    P_predicted : array-like
        Prior state covariance for all :math:`t` 0 to ntimesteps, given observations from 0 to the current time
    x_filtered: array-like
         Posterior state means for all :math:`t` 0 to `ntimesteps`, given observations from 0 to the current time :math:`t`
    P_filtered : array-like
        Posterior state covariance for all :math:`t` 0 to `ntimesteps`, given observations from 0 to the current time :math:`t`
    z :  array-like
        All observations from 0 to ntimesteps, of shape `(ntimesteps, nmatrices, ninputs)`

    Returns
    -------
    x_smooth : (ntimesteps, nmatrices, nstates) array_like
        The posterior smoothed state for each time from 0 to `ntimesteps`, computed using all observations
    x_smooth: (ntimesteps, nmatrices, nstates, nstates)
        The posterior smoothed state covariance for each time from 0 to `ntimesteps`, computed using all observations
    L : (ntimesteps, nmatrices, nstates, nstates) array-like
        The Kalman correction matrix for all steps
    """
    ntimesteps, nmatrices = get_observations_shape(z)
    nstates = get_state_size(F)

    x_smooth = np.zeros((ntimesteps, nmatrices, nstates))
    P_smooth = np.zeros((ntimesteps, nmatrices, nstates, nstates))
    L = np.zeros((ntimesteps - 1, nmatrices, nstates, nstates))
    # set mean and covariance at the end to the  forward filtered data to start the smoother
    x_smooth[-1] = x_filtered[-1]
    P_smooth[-1] = P_filtered[-1]
    # Run the smoother backwards
    for t in reversed(range(ntimesteps - 1)):
        F = pick_nth_step(F, t)
        x_smooth[t], P_smooth[t], L[t] = kalman_smoothing_step(F, x_filtered[t], P_filtered[t], x_predicted[t + 1],
                                                               P_predicted[t + 1], x_smooth[t + 1], P_smooth[t + 1])
    return x_smooth, P_smooth, L


def m_step_r(H, x_smooth, P_smooth, z):
    """
    Use the EM algorithm to estimate the observation covariance matrix given smoothed states and transition matrices.
    This part implements the M-step for the measurement covariance matrix.

    Parameters
    -------
    H : array-like
        Observation matrix with shapes  `(ntimesteps, nmatrices, noutputs, nstates)` or `(nmatrices, noutputs, nstates)`, relating the state at time :math:`t` with the output.
    x_smooth : array-like
         Posterior smoothed state for each time from 0 to `ntimesteps`, computed using all observations
    P_smooth : array-like
         Posterior smoothed state for each time from 0 to `ntimesteps`, computed using all observations

    Returns
    -------
    R_est : array-like
        Estimate observation covariance matrix

    """
    ntimesteps, nmatrices = get_observations_shape(z)
    noutputs = get_output_size(H)

    R_est = np.zeros((nmatrices, noutputs, noutputs))
    for t in range(ntimesteps):
        H = pick_nth_step(H, t)
        z_cur = pick_nth_step(z, t, ndims=1)
        x_cur = pick_nth_step(x_smooth)
        P_cur = pick_nth_step(P_smooth)
        residuals = (z_cur - matrix_vector_product(H, x_cur))
        R = tensor_outer(residuals)  # residual covariance
        H_prime = matrix_matrix_product(H, matrix_matrix_product(P_cur), transpose_tensor(H))
        R_est += R + H_prime
    return R_est / ntimesteps


def m_step_q(F, x_smooth, P_smooth, L):
    """
    Use the EM algorithm to estimate the observation covariance matrix given smoothed states and transition matrices.
    This part implements the M-step for the state transition covariance matrix.

    Parameters
    ----------
    F : array-like
        State transition matrix with shapes  `(ntimesteps, nmatrices, nstates, nstates)` or `(nmatrices, nstates, nstates)`, transition from :math:`t` to :math:`t+1` or from :math:`t_i` to :math:`t_{i + 1}` for time varying system
    x_smooth : array-like
         Posterior smoothed state for each time from 0 to `ntimesteps`, computed using all observations
    P_smooth : array-like
         Posterior smoothed state for each time from 0 to `ntimesteps`, computed using all observations
    L : array-like
        Kalman Gain matrix for each time :math:`t`

    """

    ntimesteps, nmatrices, nstates = x_smooth.shape
    Q_est = np.zeros((nmatrices, nstates, nstates))
    for t in range(ntimesteps - 1):
        L_cur = pick_nth_step(L, t, ndims=2)
        F_cur = pick_nth_step(F, t, ndims=2)
        P_cur = pick_nth_step(P_smooth, t, ndims=2)  # current smoothed covariance
        P_next = pick_nth_step(P_smooth, t + 1, ndims=2)  # next smoothed covariance
        x_s = pick_nth_step(x_smooth, t + 1, ndims=1)  # Next smoothed mean
        x_s_p = matrix_vector_product(F_cur, x_s)  # Propagated current smoothed mean
        residuals = (x_s - x_s_p)  # residual between propagated smoothed mean and current mean
        P_transf = matrix_matrix_product(matrix_matrix_product(P_next, transpose_tensor(L_cur)),
                                         transpose_tensor(F_cur))
        P_transf_1 = matrix_matrix_product(matrix_matrix_product(F_cur, P_cur), transpose_tensor(F_cur))
        Q_est += tensor_outer(residuals) + P_transf + transpose_tensor(P_transf) + P_transf_1
    Q_est /= ntimesteps
    return Q_est


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
    along  the second dimension.

    Parameters
    ----------
    F : array-like
        State transition matrix from :math:`t` to :math:`t+1`. Can be a sequence of matrices of length `ntimesteps` if the state
        transition matrix varies over time. Shape must be `(ntimesteps, nmatrices, nstates, nstates)`
    Q :  array-like
        Transition covariance (model uncertainity) matrix for the system. Must have shape `(nmatrices, nstates, nstates)`
    H : array-like
        Observation matrix to compute observation from state at time :math:`t`. Must have shapes `(ntimesteps, nmatrices, noutputs, nstates)` or `(nmatrices, noutputs, nstates)` .
    R :  array-like
        Observation covariance matrix. Must be an array of shape `(nmatrices, noutputs, noutputs)`
    x0 :  array-like optional
        Initial state mean at time 0
    P : array-like
        Initial state covariance at time 0
    """

    def __init__(self, ninputs=0, F=None, B=None, H=None, R=None, Q=None, x0=None,
                 P_0=None):

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
        if P_0 is not None:
            self.P_0 = np.array(P_0)
        else:
            self.P_0 = np.eye(self.nstates)

    def filter(self, observations):
        """
        Appy the Kalman Filter to estimate
        the state given the observations up to time :math:`t` for :math:`t` between 0 and :math:`ntimesteps-1`

        Parameters
        ----------
        observations : array-like
         Observations between 0 and `ntimesteps - 1`. Can have either shape `(ntimesteps, noutputs)` or `(ntimesteps, nmatrices, noutputs)` if\
         the filter is to be run on multiple matrices simultaneously.

        Returns
        -------
        x_filtered : array-like
            Posterior state mean for times :math:`t` between 0 and :math:`ntimesteps -1`
        P_filtered : array-like
            Posterior state covariance for times :math:`t` between 0 and :math:`ntimesteps -1`

        """

        (_, _, _, x_filtered, P_filtered) = filter(self.F, self.Q, self.H, self.R, self.x_0, self.P_0, observations)
        return x_filtered, P_filtered

    def smooth(self, observations):
        """
        Apply the Kalman Smoother to estimate the posterior state mean and covariance at all times :math:`t` given all observations between 0 and `ntimesteps`

        Parameters
        ----------
        observations : array-like
             Observations between 0 and `ntimesteps - 1`. Can have either shape `(ntimesteps, noutputs)` or `(ntimesteps, nmatrices, noutputs)` if\
         the filter is to be run on multiple matrices simultaneously.

        Returns
        -------

        """
        (x_predicted, P_predicted, K, x_filtered, P_filtered) = self.filter(observations)
        (x_smooth, P_smooth, L) = smooth(self.F, x_predicted, P_predicted, x_filtered, P_filtered, observations)
        return x_smooth, P_smooth

    # def em_observation_covariance(self, observations, transition_matrices):
    #     ntimesteps, nmatrices = get_observations_shape(observations)
    #     for t in range(ntimesteps):
    #         F = pick_nth_step(transition_matrices, t)
    #
    # def output(self, H=None):
    #     # System output
    #     H = avoid_none(H, self.H)
    #     return matrix_vector_product(H, self.x)
    #
    # def innovation(self, z, H=None):
    #     # Residual: measurement - output
    #     H = avoid_none(H, self.H)
    #     return z - self.output(H=H)
    #
    # def predict(self, u=0, F=None, B=None, Q=None):
    #     """
    #     Predict next filter state and its covariance
    #     Parameters
    #     ----------
    #     u
    #     F
    #     B
    #     Q
    #
    #     Returns
    #     -------
    #
    #     """
    #     # B = avoid_none(B, self.B)
    #     F = avoid_none(F, self.F)
    #     Q = avoid_none(Q, self.Q)
    #     # Compute next state
    #     if B is not None:
    #         control_input = matrix_vector_product(B, u)
    #     else:
    #         control_input = np.zeros(self.x.shape)
    #     self.x = matrix_vector_product(F, self.x) + control_input
    #     # Compute state covariance
    #     self.P = matrix_matrix_product(matrix_matrix_product(self.F, self.P), transpose_tensor(F).conj()) + Q
    #     # self.P = F.tensordot(self.P).tensordot(F.T.conj()) + Q
    #
    # def update(self, z, R=None, H=None):
    #
    #     R = avoid_none(R, self.R)
    #     H = avoid_none(H, self.H)
    #
    #     # Innovation
    #     y = self.innovation(z, H=H)
    #     # Residual covariance
    #     S = matrix_matrix_product(matrix_matrix_product(H, self.P), transpose_tensor(H).conj()) + R
    #     # Kalman gain
    #     K = matrix_matrix_product(matrix_matrix_product(self.P, transpose_tensor(H).conj()), special_inv(S))
    #     # K = self.P.dot(H.T.conj()).dot(special_inv(S))
    #     self.x = self.x + matrix_vector_product(K, y)
    #     P_new = matrix_matrix_product((special_eye(self.P) - matrix_matrix_product(K, H)), self.P)
    #     self.P = P_new

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

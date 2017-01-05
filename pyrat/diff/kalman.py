# Module for kalman filtering in python. Freely based on pykalman

import pickle

import numpy as np


def plot_state_and_variance(ax,t, x,P,**kwargs):
    ax.plot(t, x)
    ax.fill_between(t, x-P/2, x+P/2, facecolor='gray')

def special_inv(M):
    if np.isscalar(M):
        return 1 / M
    else:
        return np.linalg.inv(M)


def noise_fun(m, sigma, prng=None):
    #Allows to use an external prngs
    if prng is None:
        fun = np.random
    else:
        fun = prng
    if sigma.ndim == 1:#Scalar
        m + fun.randn(*m.shape) * np.sqrt(sigma)
    elif sigma.ndim == 2:#Same matrix for all pixels
        return  m + fun.multivariate_normal(np.zeros(sigma.shape[0]), sigma)
    elif sigma.ndim == 3:#A different matrix for each pixel
        return np.array([m[i] + fun.multivariate_normal(np.zeros(sigma[i].shape[0]), sigma[i]) for i in range(sigma.shape[0])])


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


def kalman_smoothing_step(F, x_filtered, P_filtered, x_predicted, P_predicted, x_smooth, P_smooth):
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
    L = matrix_matrix_product(matrix_matrix_product(P_filtered, transpose_tensor(F)), special_inv(P_predicted))
    x_smooth = x_filtered + matrix_vector_product(L, (x_smooth - x_predicted))
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
        x_filtered[t], P_filtered[t], K[t] = kalman_update_step(x_predicted[t], P_predicted[t], z_cur, H,
                                                                R)  # update
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

#Here we have the M (maximization steps) to estimate the various system parameters,
#As described in  "D. Barber, “Bayesian Reasoning and Machine Learning,” Mach. Learn., p. 646, 2011.

def m_step_r(H, x_smooth, P_smooth, z):
    """

    M step of the EM algorithm to estimate :math:`\mathbf{R}`, the data covariance matrix.


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
        Estimate observation covariance matrix after the M-step

    Notes
    -----
    This part implements the M-step for the measurement covariance matrix,called :math:`\mathbf{\Sigma_{v}}` in [1]_, eq 24.5.13  :


    References
    ----------
    .. [1] D. Barber, “Bayesian Reasoning and Machine Learning,” Mach. Learn., p. 646, 2011.

    """
    ntimesteps, nmatrices = get_observations_shape(z)
    noutputs = get_output_size(H)

    R_est = np.zeros((nmatrices, noutputs, noutputs))
    for t in range(ntimesteps):
        H = pick_nth_step(H, t)
        z_cur = pick_nth_step(z, t, ndims=1)
        x_cur = pick_nth_step(x_smooth, t, ndims=1)
        P_cur = pick_nth_step(P_smooth, t)
        residuals = (z_cur - matrix_vector_product(H, x_cur))
        R = tensor_outer(residuals)  # residual covariance
        H_prime = matrix_matrix_product(H, matrix_matrix_product(P_cur, transpose_tensor(H)))
        R_est += (R + H_prime)
    return R_est / ntimesteps


def m_step_q(F, x_smooth, P_smooth, L):
    """
    Use the EM algorithm to estimate the observation covariance matrix given smoothed states and transition matrices.
    This part implements the M-step for the state transition covariance matrix:


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


def m_step_x0(x_smooth):
    """
    Uses the EM algorithm to estimate the initial state mean by maximizing the the log-likelihood of the observations given the means.
    This function implements the M-step for the mean
    Parameters
    ----------
    x_smooth : array-like
        Smoothed posterior state mean given all observations

    Returns
    -------

    """
    return x_smooth[0]


# def m_step_P_0(x_0, x_smooth, P_smooth):
#     """
#     Uses the EM algorithm to estimate the initial state covariance.
#     This function performs the M-step:
#     .. math::
#
#         \Sigma_0 = \mathbb{E}[x_0, x_0^T] - mu_0 \mathbb{E}[x_0]^T
#                    - \mathbb{E}[x_0] mu_0^T + mu_0 mu_0^T
#     Parameters
#     ----------
#     x_0 : array-like
#         The initial state mean
#     x_smooth :
#         Smoothed posterior state mean given all observations
#     P_smooth
#          Smoothed posterior state covariance given all observations
#     Returns
#     -------
#
#     """

def m_step_all(x_smooth, P_smooth, L, z, F=None, H=None, Q=None, R=None, x_0=None, P_0=None):
    """
    Calls the EM M step to estimate the variables that are set
    to none in the Keyword arguments
    Parameters
    ----------
    F : array-like
        State transition
    H
    L
    x_smooth
    P_smooth
    Q
    R
    x0
    P_0

    Returns
    -------

    """

    if Q is None:
        Q = m_step_q(F, x_smooth, P_smooth, L)
    if R is None:
        R = m_step_r(H, x_smooth, P_smooth, z)

    return (F, H, Q, R, x_0, P_0)


def check_shape_compatilibty(arrays_and_indices, default):
    candidates = []
    for array, dim in arrays_and_indices:
        if array is not None:
            candidates.append(np.array(array).shape[dim])
    if default is not None:
        candidates.append(default)
    if len(candidates) == 0:
        return 1
    elif np.array_equal(candidates, candidates):
        return candidates[0]
    else:
        raise ValueError('The shape of the provided arrays are not consistent')


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
        transition matrix varies over time. Shape must be `(ntimesteps, nmatrices, nstates, nstates)` or `(nmatrices, nstates, nstates)`
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

    def __init__(self, F=None, H=None, R=None, Q=None, x_0=None,
                 P_0=None, nstates=None, noutputs=None, ):
        self.F = F
        self.R = R
        self.H = H
        self.Q = Q
        self.x_0 = x_0
        self.P_0 = P_0
        self._ns = nstates
        self._no = noutputs
        self._setdefaults()


    def _setdefaults(self):
        if self.F is None:
            self.F = np.array(np.eye(self.nstates), ndmin=3)
        if self.H is None:
            self.H = np.array(np.eye(self.noutputs, self.nstates), ndmin=3)
        if self.R is None:
            self.R = np.array(np.eye(self.noutputs), ndmin=3)
        if self.Q is None:
            self.Q = np.array(np.eye(self.nstates, self.nstates), ndmin=3)
        if self.x_0 is None:
            self.x_0 = np.array(np.zeros(self.nstates), ndmin=3)
        if self.P_0 is None:
            self.P_0 = np.array(np.eye(self.nstates), ndmin=3)



    @property
    def nstates(self):
        return check_shape_compatilibty(
            (
                (self.F, -1),
                (self.H, -2),
                (self.Q, -1),
                (self.x_0, -1),
                (self.P_0, -1)
            ), self._ns
        )

    @property
    def noutputs(self):
        return check_shape_compatilibty(
            (
                (self.H, -1),
                (self.R, -1),
            ), self._no
        )

    def sample(self, ntimesteps, x_0=None, seed=None):
        """
        Sample a sequence of observations :math:`y` and states :math:`x` betewen 0 and `ntimesteps`

        Parameters
        ----------
        ntimesteps : integer
            The number of steps to sample
        x_0 : optional
            Initial state. If not selected, the state is generated using the filters initial state
            mean and covariance

        Returns
        -------
            observations : array-like
                The observations
            states : array-like
                The sequence of states

        """
        # states = np.zeros(fnma)

        #set random state
        if seed is not None:
            prng = np.random.RandomState(seed)

        states = []
        outputs = []
        if x_0 is None:
            x_0 = noise_fun(self.x_0, self.P_0)
        for t in range(ntimesteps):
            if t == 0:
                states.append(np.array(x_0,ndmin=2))#ndmin to ensure that the initial state has the form 'nmat x nstates'
            else:
                F = pick_nth_step(self.F, t - 1, ndims=2)
                Q = pick_nth_step(self.Q, t - 1, ndims=2)
                R = pick_nth_step(self.R, t - 1, ndims=2)
                H = pick_nth_step(self.H, t - 1, ndims=2)
                states.append(noise_fun(matrix_vector_product(F, states[t-1]), Q,  prng=prng))
                outputs.append(noise_fun(matrix_vector_product(H, states[t]), R,  prng=prng))
        return np.array(states), np.array(outputs)

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
        (x_predicted, P_predicted, K, x_filtered, P_filtered) = filter(self.F, self.Q, self.H, self.R, self.x_0, self.P_0, observations)
        (x_smooth, P_smooth, L) = smooth(self.F, x_predicted, P_predicted, x_filtered, P_filtered, observations)
        return x_smooth, P_smooth, L

    def EM(self, observations, EM_variables=['R', 'Q'], niter=10):
        """
        Estimate filter parameters using the EM algorithm.

        Parameters
        ----------
        observations : array-like
            Observations between 0 and `ntimesteps - 1`. Can have either shape `(ntimesteps, noutputs)` or `(ntimesteps, nmatrices, noutputs)` if\
             the filter is to be run on multiple matrices simultaneously.
        variables : iterable of strings
            The EM algorithm is only run for the variable specified in 'variables'

        Returns
        -------
            KalmanFilter : KalmanFilter
                object with the parameters set in `variable` estimated using `observations`

        """
        # Determine which variables are to be estimated
        all_vars = {'F': self.F, 'H': self.H, 'Q': self.Q, 'R': self.R,
                    'x_0': self.x_0, 'P_0': self.P_0
                    }
        for variable in EM_variables:
            if variable in all_vars:
                all_vars[variable] = None
        # Iterate EM
        for i in range(niter):
            # Smooth
            x_smooth, P_smooth, L = self.smooth(observations)

            # M-step
            (self.F, self.H, self.Q, self.R, self.x_0, self.P_0) = m_step_all(x_smooth, P_smooth, L, observations, **all_vars)

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

            # @property
            # def F(self):
            #     return self._F
            #
            # @F.setter
            # def F(self, value):
            #     if np.isscalar(value) and self.nstates == 1:
            #         self._F = value
            #     elif value.shape[-2] == value.shape[-1] == self.nstates:
            #         self._F = value
            #     else:
            #         raise Exception("F is not of shape {}X{} or scalar".format(self.nstates, self.nstates))
            #
            # @property
            # def B(self):
            #     return self._B
            #
            # @B.setter
            # def B(self, value):
            #     if np.isscalar(value) and self.nstates == 1:
            #         self._B = value
            #     elif value.shape[-1] == self.nstates and self.ninputs < 2:
            #         self._B = value
            #     elif value.shape[-1] == self.nstates and value.ndim > 1 and value.shape[-1] == self.ninputs:
            #         self._B = value
            #     else:
            #         raise Exception("B is not of shape {}X{} or scalar".format(self.nstates, self.ninputs))
            #
            # @property
            # def H(self):
            #     return self._H
            #
            # @H.setter
            # def H(self, value):
            #     if np.isscalar(value) and self.nstates == 1 and self.noutputs == 1:
            #         self._H = value
            #     elif value.shape[-1] == self.nstates and self.noutputs == 1:
            #         self._H = value
            #     elif value.shape[-2] == self.noutputs and value.shape[-1] == self.nstates:
            #         self._H = value
            #     else:
            #         raise Exception("H is not of shape {}X{} or scalar".format(self.noutputs, self.nstates))
            #
            # @property
            # def P_0(self):
            #     return self._P
            #
            # @P_0.setter
            # def P_0(self, value):
            #     if np.isscalar(value) and self.nstates == 1:
            #         self._P = value
            #     elif value.shape[-2] == value.shape[-1] == self.nstates:
            #         try:  # only accept positive definite matrices
            #             isPSD(value)
            #         except np.linalg.LinAlgError:
            #             raise np.linalg.LinAlgError("P is not positive definite, cannot be used as a prior covariance matrix")
            #         self._P = value
            #     else:
            #         raise np.linalg.LinAlgError("P is not positive definite, cannot be used as a prior covariance matrix")
            #
            # @property
            # def Q(self):
            #     return self._Q
            #
            # @Q.setter
            # def Q(self, value):
            #     if np.isscalar(value) and self.nstates == 1:
            #         self._Q = value
            #     elif value.shape[-2] == value.shape[-1] == self.nstates:
            #         try:  # only accept positive definite matrices
            #             isPSD(value)
            #         except np.linalg.LinAlgError:
            #             raise np.linalg.LinAlgError("Q is not positive definite, cannot be used as a state covariance matrix")
            #         self._Q = value
            #     else:
            #         raise np.linalg.LinAlgError("Q is not positive definite, cannot be used as a state covariance matrix")
            #
            # @property
            # def R(self):
            #     return self._R
            #
            # @R.setter
            # def R(self, value):
            #     if np.isscalar(value) and self.nstates == 1 and self.noutputs == 1:
            #         self._Q = value
            #     elif value.shape[-2] == self.noutputs and self.noutputs == 1:
            #         self._Q = value
            #     elif value.shape[-2] == self.noutputs and value.shape[-1] == self.noutputs and self.noutputs > 1:
            #         try:  # only accept positive definite matrices
            #             isPSD(value)
            #         except np.linalg.LinAlgError:
            #             raise np.linalg.LinAlgError("R is not positive definite, cannot be used as a state covariance matrix")
            #         self._R = value
            #     else:
            #         raise TypeError("R is not of shape {}X{} or scalar".format(self.noutputs, self.noutputs))
            #
            # @property
            # def x_0(self):
            #     return self._x0
            #
            # @x_0.setter
            # def x_0(self, value):
            #     if np.isscalar(value) and self.nstates == 1 or \
            #                     value.shape[-1] == self.nstates:
            #         self._x0 = value

import numpy as np

class UniformMotion():
    """
    This class generates simple test data
    for an uniform linear motion with fixed initial velocity and
    initial position :math:`x_0=0`. Both states can be observed.
    """
    def __init__(self):
        self.initial_velocity = 5
        self.dt = 1
        self.initial_position = 0
        self.P_0 = np.atleast_2d(np.eye(2) * 1e-2)
        self.F = np.array([[1, self.dt], [0, 1]])
        self.H = np.array([[1, 0], [0, 1]])
        self.R = np.atleast_2d(np.eye(2) * 1e-2)
        self.Q = np.eye(2) * 1e-4

class RealData():
    """
    This class loads GPRI data and transition matrices
    computed from the 2015 Bisgletscher interferograms.
    It is mostly used to test the "tensorial" Kalman Filter that operates on several pixels at once.
    """
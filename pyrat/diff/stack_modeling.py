import numpy as _np
import pyrat.fileutils.gpri_files as _gpf
import scipy.linalg as _la


def F_model(dt, order=1):
    # F matrix for variable dt\
    # first line
    # first_line = [dt**order/(order) for order in order]
    #
    # F_m = _np.eye(order)
    F_m = _np.array([[1, dt], [0, 1]])
    return F_m


def F_aug_slc_stack(times, transition_model=F_model):
    """
    Constructs the augmented state transition
    model for a sequence of states stacked in a vector,
    where the transition between each state and
    the next is given as `transition_model(times[i]-times[i-1])`
    Parameters
    ----------
    slc_times
    transition_model

    Returns
    -------

    """
    F_aug = []
    t_start = times[0]
    for t in times[::]:
        dt = t - t_start
        F_block = transition_model(dt)
        F_aug.append(F_block)
        t_start = t
    return _la.block_diag(*F_aug)


def H_ifgram(lam, nstates=2):
    """
    Constructs the observation matrix for
    unwrapped interferogram given a state space
    model with `nstates` states where the first
    state is the displacement.
    Parameters
    ----------
    lam
    nstates

    Returns
    -------

    """
    H = _np.zeros(nstates)
    H[0] = 4 * _np.pi / lam


def H_aug_slc_stack(n_slc, H_single):
    return _la.block_diag([H_single, ] * n_slc)


class LinearSystemFromStack:
    def __init__(self, stack):
        self.stack = stack
        t_vec = [t.start_time for t in self.stack.slc_tab]
        self.F = F_aug_slc_stack(stack.t_vec)
        self.H = _np.linalg.dot(self.stack.itab.to_incidence_matrix(),
                                H_aug_slc_stack(H_ifgram(_gpf.lam(self.stack.slc_tab[0].radar_frequency))))
        #Flatten the interferograms in the first dimension
        self.z = [stack[i].reshape(-1, stack[i].shape[-1]) for i in len(stack.stack)]

class
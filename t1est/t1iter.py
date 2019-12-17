'''Try an iterative method.'''

import logging

import numpy as np

from .phase_correction import _phase_correction

def t1iter(
        x, t, T10=1, A0=1, B0=-2, time_axis=-1, alpha=1, tol=1e-4,
        maxiter=100, molli=False):
    '''Try an iterative approach.

    Parameters
    ----------
    x : array_like
        Measurements of the inversion recovery signal.
    t : array_like
        Inversion times (in sec).
    T10, A0, B0 : floats or array_like, optional
        Initial estimates for T1 (in sec), A, and B.
    time_axis : int, optional
        Axis that holds the time data.
    alpha : float, optional
        Step-size for gradient descent updates to T1 estimate.
    tol : float, optional
        Break out of loop when abs(dT1) is below tol.
    maxiter : int, optional
        The maximum number of iterations to perform.
    molli : bool, optional
        Do MOLLI correction.

    Returns
    -------
    T1, A, B : floats
        Estimates for parameters of signal equation.
    niter
        Number of iterations actually evaluated.
    err
        The derivative updates for T1 as a function of iteration.

    Notes
    -----
    Uses a the following signal model:

        s(t) = A + B exp(-t/T1)

    Notice the sign convention of B is different than that used in
    t1est.t1est().
    '''

    # Put time in the back
    assert x.ndim == 3, 'x should have xy+t dimensions!'
    x = np.moveaxis(x, time_axis, -1)

    # Do some sanity checks
    if isinstance(T10, np.ndarray):
        assert T10.shape == x.shape[:-1]
    else:
        T10 = np.ones(x.shape[:-1])*T10
    if isinstance(A0, np.ndarray):
        assert A0.shape == x.shape[:-1]
    else:
        A0 = np.ones(x.shape[:-1])*A0
    if isinstance(B0, np.ndarray):
        assert B0.shape == x.shape[:-1]
    else:
        B0 = np.ones(x.shape[:-1])*B0

    # Do phase correction so we can do a phase sensitive recon
    x = _phase_correction(x)

    # Initialize estimates
    T1, A, B = T10, A0, B0
    N = len(t)
    ii = 0

    dT1s = np.zeros(maxiter)
    amaxdT1 = np.inf
    while amaxdT1 > tol and ii < maxiter:

        # Least squares update for A, B
        etT1 = np.exp(-t[None, None, :]/T1[..., None])
        A = 1/N*np.sum(x - B[..., None]*etT1, axis=-1)
        B = np.sum(etT1*(x - A[..., None]), axis=-1)/np.sum(
            np.exp(-2*t[None, None, :]/T1[..., None]), axis=-1)

        # Do a gradient descent update step for T1
        dT1 = np.sum(
            -2*B[..., None]*t[None, None, :]*etT1*(
                x - A[..., None] - B[..., None]*etT1),
            axis=-1)/T1**2
        amaxdT1 = np.max(np.abs(dT1))
        dT1s[ii] = amaxdT1
        T1 -= alpha*dT1

        ii += 1

    # Remove unused entries in dT1s
    if ii < maxiter:
        dT1s = dT1s[..., :ii]
    else:
        logging.warning(
            'Maximum number of iterations was reached! '
            'Estimate not within tol!')

    if molli:
        logging.info('Doing MOLLI correction!')
        T1 = T1*(B/A - 1)

    return(T1, A, B, ii, dT1s)

if  __name__ == '__main__':
    pass

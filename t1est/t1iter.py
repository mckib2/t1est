'''Try an iterative method.'''

import logging

import numpy as np

def t1iter(y, t, T10, A0=1, B0=-2, alpha=1, tol=1e-4, maxiter=100):
    '''Try an iterative approach.

    Parameters
    ----------
    y : array_like
        Measurements of the inversion recovery signal.
    t : array_like
        Inversion times (in sec).
    T10, A0, B0 : floats
        Initial estimates for T1 (in sec), A, and B.
    alpha : float
        Step-size for gradient descent updates to T1 estimate.
    tol : float
        Break out of loop when abs(dT1) is below tol.
    maxiter : int
        The maximum number of iterations to perform.

    Returns
    -------
    T1, A, B : floats
        Estimates for parameters of signal equation.
    niter
        Number of iterations actually evaluated.
    err
        The derivative updates for T1 as a function of iteration.
    '''

    # Initialize estimates
    T1, A, B = T10, A0, B0
    N = len(t)
    ii = 0
    dT1 = np.inf

    dT1s = np.zeros(maxiter)
    while np.abs(dT1) > tol and ii < maxiter:

        # Least squares update for A, B
        A = 1/N*np.sum(y - B*np.exp(-t/T1))
        B = np.sum(np.exp(-t/T1)*(y - A))/np.sum(np.exp(-2*t/T1))

        # Do an update step for T1
        dT1 = np.sum(
            -2*B*t*np.exp(-t/T1)*(y - A - B*np.exp(-t/T1)))/T1**2
        dT1s[ii] = dT1
        T1 -= alpha*dT1

        ii += 1

    # Remove unused entries in dT1s
    if ii < maxiter:
        dT1s = dT1s[:ii]
    else:
        logging.warning(
            'Maximum number of iterations was reached! '
            'Estimate not within tol!')

    return(T1, A, B, ii, dT1s)

if  __name__ == '__main__':
    pass

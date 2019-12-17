'''Try an iterative method.'''

import logging

import numpy as np
from tqdm import tqdm

from .phase_correction import _phase_correction

def t1iter(
        x, t, mask=None, T10=1, A0=1, B0=-2, time_axis=-1, alpha=1,
        tol=1e-4, maxiter=100, molli=False):
    '''Try an iterative approach.

    Parameters
    ----------
    x : array_like
        Measurements of the inversion recovery signal.
    t : array_like
        Inversion times (in sec).
    mask : array_like or None, optional
        Location of pixels to do mapping.
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
    norms
        The norm of derivative updates for T1 as a function of
        iteration.

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
    if np.iscomplexobj(x):
        logging.info(
            'Doing phase correction for complex-valued input')
        x = _phase_correction(x)

    # Initialize estimates
    T1, A, B = T10, A0, B0
    N = len(t)
    ii = 0

    # Only do the vales we need
    if mask is not None:
        orig_shape = x.shape[:]
        x = np.reshape(x, (-1, x.shape[-1]))

        # Get only pixels in mask
        mask = mask.flatten()
        x = x[mask, :]
        T1 = T1.flatten()[mask][None, :]
        A = A.flatten()[mask][None, :]
        B = B.flatten()[mask][None, :]

        # Do for initial estimates just in case
        A0 = A0.flatten()[mask][None, :]
        B0 = B0.flatten()[mask][None, :]

    # Normalize data
    x /= np.linalg.norm(x)

    # Do the iterative recon:
    norms = np.zeros(maxiter)
    norm = np.inf
    pbar = tqdm(total=maxiter, leave=False)
    np.seterr(over='raise', divide='raise')
    while norm > tol and ii < maxiter:

        try:
            # Compute common vale
            etT1 = np.exp(-t[None, None, :]/T1[..., None])

            # Least squares update for A, B
            A = 1/N*np.sum(x - B[..., None]*etT1, axis=-1)

            B = np.sum(etT1*(x - A[..., None]), axis=-1)/np.sum(
                np.exp(-2*t[None, None, :]/T1[..., None]), axis=-1)

        except FloatingPointError:
            logging.warning('Overflow error! Breaking out early!')
            T1 = np.nan_to_num(T1)
            A = np.nan_to_num(A)
            B = np.nan_to_num(B)
            break

        # Do a gradient descent update step for T1
        dT1 = np.sum(
            -2*B[..., None]*t[None, None, :]*etT1*(
                x - A[..., None] - B[..., None]*etT1),
            axis=-1)/T1**2
        dT1[np.abs(dT1) > 10] = 10 # remove ridiculous values
        dT1 = np.nan_to_num(dT1)
        norm = np.linalg.norm(dT1)
        dT1 /= norm
        T1 -= alpha*dT1

        # remember stopping condition:
        norms[ii] = norm

        ii += 1
        pbar.update(1)
    pbar.close()

    # Remove unused entries in norms
    if ii < maxiter:
        norms = norms[..., :ii]
    else:
        logging.warning(
            'Maximum number of iterations was reached! '
            'Estimate not within tol!')

    # Do the MOLLI correction if user asked for it
    if molli:
        logging.info('Doing MOLLI correction!')
        T1 = T1*(np.abs(B/A) - 1)

    # Move data back from masked shape
    if mask is not None:
        tmp = np.zeros(np.prod(orig_shape[:-1]))
        tmp[mask] = T1.squeeze()
        T1 = tmp
        T1 = np.reshape(T1, orig_shape[:-1])

        tmp = np.zeros(np.prod(orig_shape[:-1]))
        tmp[mask] = A.squeeze()
        A = tmp
        A = np.reshape(A, orig_shape[:-1])

        tmp = np.zeros(np.prod(orig_shape[:-1]))
        tmp[mask] = B.squeeze()
        B = tmp
        B = np.reshape(B, orig_shape[:-1])

    return(T1, A, B, ii, norms)

if  __name__ == '__main__':
    pass

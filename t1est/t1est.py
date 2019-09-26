'''Simple T1 estimation.'''

from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm

def t1est(
        x, TIs, time_axis=-1, mask=None, method='lm', T1_bnds=None,
        chunksize=10, molli=False):
    '''Make a T1 map.

    Parameters
    ----------
    x : array_like
        Complex image space data.
    TIs : array_like
        Inversion times.
    time_axis : int, optional
        Dimension of x that holds the time points.
    mask : array_like
        In-plane mask.
    method : {'trf', 'lm'}, optional
        Trust region reflective or Lev-Mar for fitting nonlinear
        least squares. Lev-Mar is faster, trust region is more
        accurate and can use bounds.
    T1_bnds : tuple or None, optional
        Upper and lower bound for T1 to use when fitting.  Only for
        method='trf'.
    chunksize : int, optional
        Multiprocessing chunksize.
    molli : bool, optional
        Do MOLLI correction.  T1_bnds will be considered as bounds on
        apparent T1.

    Returns
    -------
    T1map : array_like
        T1 estimates the same shape as mask.
    '''

    # Move time points to the back of the bus
    x = np.moveaxis(x, time_axis, -1)

    # Sanity checks
    TIs = np.array(TIs)
    nt = x.shape[-1]
    sh = x.shape[:-1]
    assert nt == TIs.size, (
        'Number of inversion times does not match the number of '
        'time frames in x!')
    assert mask.shape == sh, 'mask should match x!'

    # Get bounds and initial guess for least squares solver
    x0 = np.ones(5)
    if method == 'trf':
        if T1_bnds is None:
            T1_bnds = (0, np.inf)
        bnds = (
            (-np.inf, -np.inf, -np.inf, -np.inf, T1_bnds[0]),
            (np.inf, np.inf, np.inf, np.inf, T1_bnds[1]))
    elif method == 'lm':
        bnds = (-np.inf, np.inf)
    else:
        raise NotImplementedError()

    # If we don't have a mask, just do everything
    if mask is None:
        mask = np.ones(sh, dtype=bool)

    # Parallelize this thing!
    ntot = np.prod(sh)
    T1map = np.zeros(ntot)
    mask = mask.flatten()
    x = np.reshape(x, (-1, nt))
    idx = np.argwhere(mask).squeeze()
    T1map = T1map.flatten()
    fitp = partial(
        _fitwrap, x=x, TIs=TIs, method=method, bnds=bnds,
        molli=molli, x0=x0)
    with Pool() as pool:
        res = list(tqdm(
            pool.imap(fitp, idx, chunksize),
            total=idx.size, leave=False))
    T1map[idx] = np.array(res)
    return np.reshape(T1map, sh)

def _fitwrap(idx, x, TIs, method, bnds, molli, x0):
    '''Picklable wrapper for _fit.'''
    return _fit(x[idx, :], TIs, method, bnds, molli, x0)

def _model(Ar, Ai, Br, Bi, T1, TI):
    '''T1 fitting model with complex coefficients.'''
    return (Ar + 1j*Ai) - (Br + 1j*Bi)*np.exp(-TI/T1)

def _obj(x, t, y):
    '''Function for least squares fitting'''
    Ar, Ai, Br, Bi, T1 = x[:]
    return np.abs(_model(Ar, Ai, Br, Bi, T1, t) - y)

def _fit(y, t, method, bnds, molli, x0):
    '''Do T1 fitting'''

    res_lsq = least_squares(
        _obj, x0, args=(t, y), method=method, bounds=bnds)
    Ar, Ai, Br, Bi, T1 = res_lsq['x'][:]

    if molli:
        return T1*(np.abs(Br + 1j*Bi)/np.abs(Ar + 1j*Ai) - 1)
    return T1

if __name__ == '__main__':
    pass

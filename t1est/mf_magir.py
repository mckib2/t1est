'''MF-MAGIR T1 fitting for MOLLI data.'''

import logging

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.interpolate import interpn
from scipy.sparse import lil_matrix
from tqdm import tqdm


def _fit(s, TI, x0, method='Nelder-Mead', bnds=((-np.inf, -np.inf, 0), (np.inf, np.inf, 5))):
    '''Use minimize(), then if fail try least_squares(method='trf').'''
    def model(x):
        return x[0]*(1 - x[1]*np.exp(-TI/x[2]))
    def fun(x):
        return model(x) - s
    def scalar_fun(x):
        return np.sqrt(np.sum(fun(x)**2))
    def molli(res):
        return res.x[2]*(res.x[1] - 1)*1000
    res = minimize(scalar_fun, x0, method=method)
    T1 = molli(res)
    x = res.x
    resid = res.fun
    if T1 < 0 or res.status != 0:
        # logger.info('Failed, trying trf() fit')
        res = least_squares(fun, x0, method='trf', bounds=bnds)
        resid2 = scalar_fun(res.x)
        if resid2 < resid:
            T1 = molli(res)
            x = res.x
            resid = resid2
    return T1, x, resid


def mf_magir(data, TI, x0_init=[1, 2, 1], mask=None, max_polarity_len=None, time_axis=-1):
    '''MF-MAGIR.

    Parameters
    ----------
    data : array_like
        Magnitude MOLLI data.
    TI : array_like
        Inversion times.
    x0_init : array_like (3,), optional
        Initial estimates of [A, B, T1*]
    mask : array_like, optional
        Boolean mask to fit. Fits entire data if None provided.
    max_polarity_len : int, optional
        Number of time sequence points to invert to try to find
        correct polarities.  If None, uses `data.shape[time_axis]//2`.
    time_axis : int, optional
        The dimension holding time.

    Returns
    -------
    T1map : array_like
        The T1 map that was fit was from the data using the model:
            S(TI) = A*(1 - B*exp{-TI/T1*})
    '''
    data = np.moveaxis(data, time_axis, -1)
    orig_sh = data.shape[:]
    nt = data.shape[-1]
    if mask is None:
        mask = np.ones(orig_sh[:-1], dtype=bool)
    elif mask.shape != orig_sh[:-1]:
        raise ValueError(f'mask should apply over spatial dimensions of data! {mask.shape} != {orig_sh[:-1]}')
    data = np.reshape(data, (-1, nt))
    data /= np.max(data.flatten())
    mask = mask.flatten()

    if max_polarity_len is None:
        max_polarity_len = nt // 2

    # fit each px in mask
    T1 = np.zeros(mask.shape)
    mask_idxs = np.argwhere(mask).squeeze()
    x0 = x0_init
    global_best_resid = np.inf
    for ii in tqdm(mask_idxs):
        # do the fit for all possible polarities and
        # choose the one with the lowest residual
        s = data[ii, :].copy()
        best_resid = np.inf
        best_x0 = None
        for jj in range(max_polarity_len):
            s[jj] *= -1
            T10, x00, resid = _fit(s, TI, x0)
            if resid < best_resid:
                # logger.info('Index %d found better polarity at %d', ii, jj)
                best_resid = resid
                best_x0 = x00
                T1[ii] = T10
        # consider the last fitting params the next inital
        if best_resid < global_best_resid:
            logger.info('New best x0: %s', x0)
            global_best_resid = best_resid
            x0 = best_x0

    return np.reshape(T1, orig_sh[:-1])


if __name__ == '__main__':
    import pathlib
    from scipy.io import loadmat

    prefix = pathlib.Path('data/john-data')
    data = loadmat(prefix / 'testData.mat')['testData'].squeeze()
    T1 = loadmat(prefix / 'testT1.mat')['testT1'].squeeze()
    TI = loadmat(prefix / 'testTI.mat')['testTI'].squeeze()

    # uncomment to reconstruct a reduced FOV:
    # # center square
    # ctr = (data.shape[-3] // 2, data.shape[-2] // 2)
    # pd = 30
    # data = data[:, ctr[0]-pd:ctr[0]+pd, ctr[1]-pd:ctr[1]+pd, :]
    # T1 = T1[:, ctr[0]-pd:ctr[0]+pd, ctr[1]-pd:ctr[1]+pd]

    # Scaling and normalization
    TI *= 1e-3
    data /= np.max(data.flatten())
    print(data.shape, T1.shape, TI.shape)

    sl = 2
    T1est = mf_magir(data[sl, ...], TI[sl, :])

    import matplotlib.pyplot as plt
    plt.subplot(1, 3, 1)
    plt.imshow(T1[sl, ...], vmin=0, vmax=4000)
    plt.title('True')
    plt.subplot(1, 3, 2)
    plt.imshow(T1est, vmin=0, vmax=4000)
    plt.title('Est')
    plt.subplot(1, 3, 3)
    resid = (T1[sl, ...] - T1est).flatten()
    plt.plot(resid)
    plt.title(f'True - Est, mean: {np.mean(resid[np.abs(resid) < 1e3])}')  # discard obvious outliers
    plt.xlabel('Flattened index')
    plt.ylabel('ms')
    plt.show()

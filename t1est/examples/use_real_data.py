'''Try T1 mapping on real data.'''

from time import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.filters import threshold_li

from t1est import t1iter

if __name__ == '__main__':

    path = 'data/FID12461.mat'
    data = loadmat(path)['kspace']
    print(data.shape)
    # data *= 1e8
    t = np.array([
        117., 257., 1172., 1282., 2172., 2325., 3174., 4189.])
    t *= 1e-3
    time_frames = [0, 5, 1, 6, 2, 7, 3, 4]
    data = data[..., time_frames, :]

    ifft = lambda x0: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        x0, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    # Put into imspace
    ims = ifft(data)

    plt_opts = {
        'vmin': 0,
        'vmax': 2.5,
    }

    # Look at only pixels that have strong enough signal
    sos = np.sqrt(np.sum(np.abs(ims)**2, axis=-2)) # across time
    thresh = threshold_li(sos)
    mask = sos > thresh/2

    # Run on each coil
    # T1s = np.zeros(data.shape[:2])
    T1s = []
    A, B = 1, -2
    for cc in range(ims.shape[-1]):

        t0 = time()
        ims0 = ims[..., cc]
        T1, A, B, niter, err = t1iter(
            ims0, t, mask=mask[..., cc], T10=1, A0=A, B0=B,
            time_axis=-1, alpha=.05, tol=1e-4, maxiter=750,
            molli=True)
        print('Took %g sec for coil %d' % (time() - t0, cc))

        # Add onto composite
        T1s.append(T1)

        # plt.semilogy(err)
        # plt.show(block=False)
        #
        # plt.imshow(T1, **plt_opts)
        # plt.title('Coil %d' % cc)
        # plt.show()

    # Average all coils together
    sos = np.sqrt(np.sum(np.abs(ims)**2, axis=-1)) # across coils
    weights = []
    for cc in range(ims.shape[-1]):
        mask0 = mask[..., cc][..., None]
        w = np.linalg.norm(
            sos*mask0 - np.abs(ims[..., cc]*mask0), axis=-1)**2
        weights.append(w)
    weights = weights[:len(T1s)]

    weights = np.moveaxis(np.array(weights), 0, -1)
    weight_sum = np.sum(weights, axis=-1)
    weights = weights/weight_sum[..., None]
    weights = np.nan_to_num(weights)
    weights = [weights[..., cc] for cc in range(weights.shape[-1])]

    T1 = np.concatenate(T1s, axis=-1)

    # T1s /= np.sum(mask, axis=-1)
    T1 = np.zeros(ims.shape[:2])
    for cc, (T10, w) in enumerate(zip(T1s, weights)):
        mask0 = mask[..., cc]
        T1[mask0] += T10[mask0]*w[mask0]

    plt.imshow(T1, **plt_opts)
    plt.show()

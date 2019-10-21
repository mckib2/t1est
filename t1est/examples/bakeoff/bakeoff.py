'''Run datasets from Ganesh and compare to Siemens' truth.'''

from time import time
from os.path import isfile

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from skimage.measure import compare_nrmse, compare_ssim
from skimage.filters import threshold_li
from skimage.morphology import convex_hull_image

from t1est import t1est
from .loader import loader

def run(
        TIs, ims, truth, mask, method, chunksize=50, filename=None,
        show=True, force=False):
    '''Run a dataset.'''

    if not isfile(filename + '.npz') or force:
        T1map = t1est(
            ims, TIs, time_axis=-1, mask=mask, method=method,
            T1_bnds=(0, 5), chunksize=chunksize, molli=True, mag=True)
        T1map = np.nan_to_num(T1map)

        # Assume that anything estimated above 5s is wrong
        T1map[np.abs(T1map) > 5] = 0

        # Assume that anything estimated below 0s is wrong
        T1map[T1map < 0] = 0

        np.savez_compressed(
            filename + '.npz',
            mask=mask,
            method=method,
            T1map=T1map)

    else:
        print('Result exists!')
        data = np.load(filename + '.npz')
        # savemat(filename + '.mat', mdict=data)
        T1map = data['T1map']


    # Some plots
    plt.figure(figsize=(10, 3))
    plt_args = {'vmin':0, 'vmax':4}
    nx, ny = 1, 3
    ax = plt.subplot(nx, ny, 1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(truth, **plt_args)
    plt.title('Truth')

    ax = plt.subplot(nx, ny, 2)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(T1map, **plt_args)
    plt.title('T1 est (%s)' % method)

    ax = plt.subplot(nx, ny, 3)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(truth - T1map, **plt_args)
    nrmse = compare_nrmse(truth, T1map)
    ssim = compare_ssim(truth, T1map)
    print('NRMSE: %g' % nrmse)
    print(' SSIM: %g' % ssim)
    plt.title('NRMSE: %g, SSIM: %g' % (nrmse, ssim))

    if filename:
        plt.savefig('%s.png' % filename)
    if show:
        plt.show()

def find_colorbar(idx):
    '''Give back location of colorbar'''

    # The colorbar appears to be in one of two positions:
    pos0 = (67, 112, 5, 16)
    pos1 = (76, 128, 4, 13)
    return [
        pos0, pos0, pos1, pos0, pos0, pos0, pos1, pos1, pos1, pos1,
        pos1, pos0, pos1, pos1, pos1, pos1][idx]

if __name__ == '__main__':

    # num_pts = 8
    num_pts = 5

    for idx in range(16):

        print('STARTING %d' % idx)
        TIs, ims, truth, name = loader(idx)

        # Pop images off of the front until we have the ones we want
        if ims.shape[-1] > num_pts:
            nt = ims.shape[-1]
            # step = int(np.ceil(nt/num_pts))
            # ind = [ii for ii in range(0, nt, step)]
            # if len(ind) == num_pts-1:
            #     ind.insert(int(len(ind)/2), int(nt/2)-1)
            # print(ind)
            # ims = ims[..., ind]
            # TIs = TIs[ind]
            # print(ims.shape, TIs.shape)

            # Use first 5 time points
            ind = [ii for ii in range(num_pts)]
            ims = ims[..., ind]
            TIs = TIs[ind]
            print(ims.shape, TIs.shape)


        t0 = time()
        sos = np.sqrt(np.sum(np.abs(ims)**2, axis=-1))
        thresh = threshold_li(sos)
        mask = convex_hull_image(sos > thresh)

        # Mask should always include Siemens' colorbar
        x, xx, y, yy = find_colorbar(idx)
        mask[x:xx, y:yy] = False

        truth = truth*mask
        print('Took %g seconds to get mask' % (time() - t0))

        run(
            TIs, ims, truth, mask, method='lm', chunksize=50,
            filename='data/results/lm_first_t%d_%s' % (num_pts, name),
            show=False, force=False)

        # run(
        #     TIs, ims, truth, mask, method='trf', chunksize=20,
        #     filename='data/results/trf_%s' % name, show=False)
